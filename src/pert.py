import copy
import logging
import math
import torch
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np


def random_init_lw(delta_dict, rho, orig_param_dict, norm="l2", adaptive=False):
    assert norm in ["l2", "linf"], f"Unknown perturbation model {norm}."

    for param in delta_dict:
        if norm == "l2":
            delta_dict[param] = torch.randn_like(delta_dict[param]).cuda()
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

    n_batches, avg_loss, avg_init_loss = 0, 0.0, 0.0
    output_str = ""

    with torch.no_grad():
        sharp_hist = []
        for _ in tqdm(range(n_iters)):
            x,y = next(iter(dataloader))
            x, y = x.cuda(), y.cuda()

            # Loss on the unperturbed model.
            init_loss = get_loss(model, loss_f, x, y)
            avg_init_loss += init_loss

            batch_loss = 0.0
            cur_loss_list = []
            for i in range(n_iters):
                delta_dict = random_init_lw(
                    delta_dict, rho, orig_param_dict, norm=norm, adaptive=adaptive
                )
                for (param_name, delta), (_, param) in zip(
                    delta_dict.items(), noisy_model.named_parameters()
                ):
                    param.data = orig_param_dict[param_name] + delta_dict[param_name]

                curr_loss = get_loss(noisy_model, loss_f, x, y)
                batch_loss += curr_loss
                cur_loss_list.append(curr_loss)
            var = np.var(cur_loss_list)
            required_iter = max(int(np.sqrt(var / 0.01)), n_iters)
            # logging.info(f"Variance of loss: {var}")
            for i in range(required_iter - n_iters):
                delta_dict = random_init_lw(
                    delta_dict, rho, orig_param_dict, norm=norm, adaptive=adaptive
                )
                for (param_name, delta), (_, param) in zip(
                    delta_dict.items(), noisy_model.named_parameters()
                ):
                    param.data = (
                        orig_param_dict[param_name] + delta_dict[param_name]
                    )

                curr_loss = get_loss(noisy_model, loss_f, x, y)
                batch_loss += curr_loss

            n_batches += 1
            avg_loss += batch_loss / max(n_iters, required_iter)
            sharp_hist.append(batch_loss / max(n_iters, required_iter))
        sharp_var = np.var(sharp_hist)
        logging.info(f"Variance of sharpness: {sharp_var}")
            

    sharpness = (avg_loss - avg_init_loss) / n_batches

    return sharpness


def eval_mls(model, train_loader, args, nonlin, model_name, n_iters=100, num_samples=256):
    mls_hist = []
    norm_mls_hist = []
    mls_sum, norm_mls_sum = 0.0, 0.0

    def cal_mls():
        img, target = next(iter(train_loader))
        img, target = img.to(args.device), target.to(args.device)
        alp = args.rho**2  # Noise variance
        # jacobian(model, img) # This takes 17 minutes to run
        noise = torch.randn(num_samples, *(img.shape[1:])).to(args.device) * math.sqrt(
            alp
        )
        out = model(img + noise)
        out = nonlin(out)
        cov = torch.cov(out.T) / alp
        assert cov.shape == (1000, 1000)
        mls = torch.linalg.matrix_norm(cov, ord='fro').cpu().item()
        norm_mls = mls * torch.norm(img.flatten()).cpu().item()
        return mls, norm_mls

    for _ in tqdm(range(n_iters)):
        mls, norm_mls = cal_mls()
        mls_hist.append(mls)
        norm_mls_hist.append(norm_mls)
        mls_sum = mls_sum + mls
        norm_mls_sum = norm_mls_sum + norm_mls

    mls_var = np.var(mls_hist)
    logging.info(f"Variance of MLS: {mls_var}")
    norm_mls_var = np.var(norm_mls_hist)
    logging.info(f"Variance of Normalized MLS: {norm_mls_var}")

    required_iter = int(mls_var / 1e3)
    required_iter = max(required_iter, n_iters)

    for _ in tqdm(range(required_iter - n_iters)):
        mls, norm_mls = cal_mls()
        mls_hist.append(mls)
        norm_mls_hist.append(norm_mls)
        mls_sum = mls_sum + mls
        norm_mls_sum = norm_mls_sum + norm_mls
    mls_avg = mls_sum / required_iter
    norm_mls_avg = norm_mls_sum / required_iter
    logging.info(f"MLS for model {model_name}: {mls_avg}")
    logging.info(f"Normalized MLS for model {model_name}: {norm_mls_avg}")
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
            sharp_sum += torch.linalg.matrix_norm(cur_sharpness, ord="fro").cpu().item()

    sharpness = sharp_sum / n_batches

    return sharpness
