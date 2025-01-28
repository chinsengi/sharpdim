import copy
import logging
import math
import torch
from tqdm import tqdm
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


def random_init_lw(delta_dict, rho, orig_param_dict, norm="l2", adaptive=False, device="cuda"):
    assert norm in ["l2", "linf"], f"Unknown perturbation model {norm}."

    for param in delta_dict:
        if norm == "l2":
            delta_dict[param] = torch.randn_like(delta_dict[param]).to(device)
        elif norm == "linf":
            delta_dict[param] = (
                2 * torch.rand_like(delta_dict[param], device="cuda") - 1
            )

    for param in delta_dict:
        param_norm_curr = orig_param_dict[param].abs() if adaptive else 1
        delta_dict[param] *= rho * param_norm_curr

    return delta_dict


def param_count(model):
    return sum(p.numel() for p in model.parameters())


# # ref: https://github.com/tml-epfl/sharpness-vs-generalization/blob/main/sharpness.py#L320
def eval_average_sharpness(
    model,
    dataloader,
    loss_f,
    n_iters=100,
    rho=1.0,
    adaptive=False,
    norm="l2",
    nonlin=None,
    device="cuda",
):
    """Average case sharpness with Gaussian noise ~ (0, rho)."""

    def get_loss(model, loss_fn, x, y):
        """Compute loss and class. error on a single batch."""
        with torch.no_grad():
            output = model(x)
            output = nonlin(output)
            if y.dim() == 1:
                targets_one_hot = torch.zeros_like(output)
                targets = targets_one_hot.scatter_(1, y.unsqueeze(1), 1)
            loss = loss_fn(output, targets)

        return loss.cpu().item()

    orig_param_dict = {
        param_name: p.clone() for param_name, p in model.named_parameters()
    }  # {param: param.clone() for param in model.parameters()}
    noisy_model = copy.deepcopy(model)

    delta_dict = {
        param_name: torch.zeros_like(param)
        for param_name, param in model.named_parameters()
    }
    logging.info(f"Named params: {len(delta_dict)}")
    logging.info(f"Params: {param_count(model)}")
    logging.info(f"rho: {rho} samples: {n_iters}")

    n_batches, loss_sum, init_loss_sum = 0, 0.0, 0.0
    sample_iters = n_iters
    pert_iters = 50
    with torch.no_grad():
        sharp_hist = []
        for _ in tqdm(range(sample_iters)):
            x, y = next(iter(dataloader))
            x, y = x.to(device), y.to(device)

            # Loss on the unperturbed model.
            init_loss = get_loss(model, loss_f, x, y)
            init_loss_sum += init_loss

            batch_loss = 0.0
            cur_loss_list = []
            for _ in range(pert_iters):
                delta_dict = random_init_lw(
                    delta_dict, rho, orig_param_dict, norm=norm, adaptive=adaptive, device=device
                )
                for (param_name, delta), (_, param) in zip(
                    delta_dict.items(), noisy_model.named_parameters()
                ):
                    param.data = orig_param_dict[param_name] + \
                        delta_dict[param_name]

                curr_loss = get_loss(noisy_model, loss_f, x, y)
                batch_loss += curr_loss
                cur_loss_list.append(curr_loss)
            std = np.std(cur_loss_list)
            required_iter = max(int(std / (rho * 10)) ** 2, pert_iters)
            required_iter = min(required_iter, 200)
            logging.warning(f"{required_iter=}")
            for _ in range(required_iter - pert_iters):
                delta_dict = random_init_lw(
                    delta_dict, rho, orig_param_dict, norm=norm, adaptive=adaptive, device=device
                )
                for (param_name, delta), (_, param) in zip(
                    delta_dict.items(), noisy_model.named_parameters()
                ):
                    param.data = orig_param_dict[param_name] + \
                        delta_dict[param_name]

                curr_loss = get_loss(noisy_model, loss_f, x, y)
                batch_loss += curr_loss
                cur_loss_list.append(curr_loss)
            n_batches += 1
            assert required_iter == len(
                cur_loss_list
            ), f"{required_iter=} {len(cur_loss_list)=}"
            loss_sum += batch_loss / required_iter
            sharp_hist.append(batch_loss / required_iter - init_loss)
            logging.info(
                f"sharpness: {batch_loss / required_iter - init_loss}")
    sharp_std = np.std(sharp_hist) / (np.sqrt(n_batches) * rho**2)
    logging.warning(f"standard deviation of sharpness: {sharp_std}")

    sharpness = (loss_sum - init_loss_sum) / (n_batches * rho**2)
    logging.warning(f"Sharpness: {sharpness}")

    # w_norm = 0.
    # for  _, param in model.named_parameters():
    # w_norm += torch.norm(param).cpu().item()**2
    high_quality = False
    # logging.warning(f"Weight-norm Sharpness: {np.sqrt(w_norm) * np.sqrt(sharpness)}")
    if abs(sharp_std / sharpness) < 0.15:
        high_quality = True
    return sharpness, high_quality
    return sharpness, sharpness * w_norm, high_quality


# adversarial MLS
def eval_mls_adv(
    model,
    train_loader,
    args,
    nonlin,
    n_iters=100,
    device="cuda",
):
    epsilon = args.rho

    mls_hist = []
    norm_mls_hist = []
    mls_sum, norm_mls_sum = 0.0, 0.0
    for _ in tqdm(range(n_iters)):
        img, target = next(iter(train_loader))
        img, target = img.to(device), target.to(device)
        orig = model(img)
        delta = torch.randn_like(img, requires_grad=True, device=device)
        # breakpoint()
        opt = optim.Adam([delta], lr=0.1)
        delta.data = delta.data * epsilon
        for _ in range(30):
            pert = model(img + delta)
            loss = -torch.norm(nonlin(orig) - nonlin(pert)) / \
                torch.norm(delta.flatten())
            opt.zero_grad()
            loss.backward()
            # breakpoint()
            opt.step()
            # logging.info(f"loss: {loss.cpu().item()}")
            # logging.info(f"norm: {-torch.norm(nonlin(orig) - nonlin(pert))}")
        mls = loss.cpu().item()**2
        norm_mls = mls * torch.norm(img.flatten()).cpu().item()**2
        mls_hist.append(mls)
        norm_mls_hist.append(norm_mls)
        mls_sum = mls_sum + mls
        norm_mls_sum = norm_mls_sum + norm_mls
    std = np.std(mls_hist) / np.sqrt(n_iters)
    logging.warning(f"Standard deviation of MLS: {std}")
    norm_std = np.std(norm_mls_hist) / np.sqrt(n_iters)
    logging.warning(f"standard deviation of Normalized MLS: {norm_std}")
    logging.warning(f"MLS for model: {mls_sum / n_iters}")
    logging.warning(f"Normalized MLS for model: {norm_mls_sum / n_iters}")

    high_quality = False
    if abs(std / (mls_sum / n_iters)) < 0.15:
        high_quality = True
    return mls_sum / n_iters, norm_mls_sum / n_iters, high_quality


def jacobian_vector_product(model, x, nonlin):
    with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False):
        x.requires_grad_(True)
        jvp = torch.autograd.functional.jvp(lambda x: nonlin(model(x)), x, v=x)[1]
    return jvp


def eval_iimls(model, train_loader, args, nonlin, n_iters=100, device="cuda"):
    model.eval()
    iimls_sum = 0.0
    iimls_hist = []

    for _ in tqdm(range(n_iters)):
        x, _ = next(iter(train_loader))
        x = x.to(device)

        # Calculate jacobian_vector_product
        jvp = jacobian_vector_product(model, x, nonlin)

        # Calculate IIMLS
        iimls = torch.sum(jvp ** 2 / torch.numel(x))

        iimls_sum += iimls.cpu().item()
        iimls_hist.append(iimls.cpu().item())
        logging.info(f"IIMLS: {iimls.cpu().item()}")

    # Calculate averages
    avg_iimls = iimls_sum / n_iters


    # Calculate standard deviations
    iimls_std = torch.tensor(iimls_hist).std().item()
    logging.warning(f"IIMLS for model: {avg_iimls}")
    logging.warning(f"Standard deviation of IIMLS: {iimls_std}")

    # Check if the estimation is high quality
    high_quality = abs(iimls_std / avg_iimls) < 0.15

    return avg_iimls, high_quality


def eval_mls(
    model, train_loader, args, nonlin, model_name, n_iters=100, num_samples=256
):
    mls_hist = []
    norm_mls_hist = []
    mls_sum, norm_mls_sum = 0.0, 0.0

    def cal_mls():
        img, target = next(iter(train_loader))
        img, target = img.to(args.device), target.to(args.device)
        alp = args.rho**2  # Noise standard deviation
        # jacobian(model, img) # This takes 17 minutes to run
        mls, norm_mls = 0.0, 0.0
        for _ in range(num_samples // 32):
            noise = torch.randn(
                32, *(img.shape[1:])).to(args.device) * math.sqrt(alp)
            out = model(img + noise)
            out = nonlin(out)
            cov = torch.cov(out.T) / alp
            assert cov.shape == (1000, 1000)
            mls += torch.linalg.matrix_norm(cov, ord="fro").cpu().item()
            norm_mls += mls * torch.norm(img.flatten()).cpu().item()
        return mls / (num_samples // 32), norm_mls / (num_samples // 32)

    for _ in tqdm(range(n_iters)):
        mls, norm_mls = cal_mls()
        mls_hist.append(mls)
        norm_mls_hist.append(norm_mls)
        mls_sum = mls_sum + mls
        norm_mls_sum = norm_mls_sum + norm_mls

    mls_std = np.std(mls_hist)
    logging.warning(f"standard deviation of MLS: {mls_std}")
    norm_mls_std = np.std(norm_mls_hist)
    logging.warning(f"standard deviation of Normalized MLS: {norm_mls_std}")

    required_iter = int(mls_std / 1e2)
    required_iter = max(required_iter, n_iters)

    for _ in tqdm(range(required_iter - n_iters)):
        mls, norm_mls = cal_mls()
        mls_hist.append(mls)
        norm_mls_hist.append(norm_mls)
        mls_sum = mls_sum + mls
        norm_mls_sum = norm_mls_sum + norm_mls
    mls_avg = mls_sum / required_iter
    norm_mls_avg = norm_mls_sum / required_iter
    logging.warning(f"MLS for model {model_name}: {mls_avg}")
    logging.warning(f"Normalized MLS for model {model_name}: {norm_mls_avg}")
    return mls_avg, norm_mls_avg


def eval_cov_sharpness(
    model,
    dataloader,
    n_iters=100,
    rho=1.0,
    norm="l2",
):
    """Average case sharpness with Gaussian noise ~ (0, rho)."""

    orig_param_dict = {
        param_name: p.clone() for param_name, p in model.named_parameters()
    }
    noisy_model = copy.deepcopy(model)

    delta_dict = {
        param_name: torch.zeros_like(param)
        for param_name, param in model.named_parameters()
    }
    logging.info(f"Named params: {len(delta_dict)}")
    logging.info(f"Params: {param_count(model)}")
    logging.info(f"rho: {rho} samples: {n_iters}")

    n_batches, sharp_sum = 0, 0.0

    model.eval()
    with torch.no_grad():
        for x, y in tqdm(dataloader):
            x, y = x.cuda(), y.cuda()

            output = torch.zeros(n_iters, 1000).cuda()
            for i in range(n_iters):
                delta_dict = random_init_lw(
                    delta_dict, math.sqrt(rho), orig_param_dict, norm=norm
                )
                for (param_name, delta), (name, param) in zip(
                    delta_dict.items(), noisy_model.named_parameters()
                ):
                    assert param_name == name
                    param.data = orig_param_dict[param_name] + delta

                output[i] = noisy_model(x)

            n_batches += 1
            cur_sharpness = torch.cov(output.T) / rho
            assert cur_sharpness.shape == (1000, 1000)
            sharp_sum += torch.linalg.matrix_norm(
                cur_sharpness, ord="fro").cpu().item()

    sharpness = sharp_sum / n_batches

    return sharpness
