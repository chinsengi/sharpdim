import logging
import math
import os
import time
from matplotlib import pyplot as plt
import numpy as np
from regex import W
import torch
import torch.nn as nn
import logging
import random

from tqdm import tqdm
from .models.vit import vit
from .models.vgg import vgg11, vgg11_big, VGG
from .models.fnn import LeNet5, fnn, lenet, lenet5, LeNet, FNN
from .models.resnet import resnet
from .data import DataLoader, load_fmnist, load_cifar10, load_imagenet
from .linalg import eigen_variance, eigen_hessian


def use_gpu(gpu_id: int = 0):
    num_of_gpus = torch.cuda.device_count()
    if num_of_gpus > 0:
        assert gpu_id < num_of_gpus
    device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")
    return device


# load a model
def load(path, model, optimizer=None):
    if os.path.exists(path):
        state = torch.load(path, map_location=torch.device("cpu"))
        model.load_state_dict(state[0])
        if optimizer is not None:
            optimizer.load_state_dict(state[1])
    else:
        raise FileNotFoundError(f"weight file not found from {path}")
        logging.warning(f"weight file not found from {path}, training from scratch")


def save_model(model, optimizer, path, filename):
    create_dir(path)
    states = [model.state_dict(), optimizer.state_dict()]
    torch.save(states, os.path.join(path, filename))


def save_npy(obj, path, filename):
    create_dir(path)
    logging.info(f"Saving to {path}/{filename}")    
    np.save(os.path.join(path, filename), obj)


def savefig(path="./image", filename="image", format="png", include_timestamp=True):
    create_dir(path)
    if include_timestamp:
        t = time.localtime()
        current_time = time.strftime("%H:%M:%S", t)
    else:
        current_time = ""
    plt.savefig(
        os.path.join(path, current_time + filename + "." + format),
        dpi=300,
        format=format,
    )
    logging.info(f"Saved figure to {path}/{current_time + filename + '.' + format}")


# create directory
def create_dir(path="./model"):
    isExist = os.path.exists(path)
    if not isExist:
        print(f"Creating directory: {path}")
        os.makedirs(path)


def load_net(network, dataset, num_classes, nonlinearity, use_layer_norm=False):
    if network == "fnn":
        return fnn(dataset, num_classes, nonlinearity, use_layer_norm)
    elif network == "vgg":
        if dataset == "fashionmnist":
            in_channel = 1
        else:
            in_channel = 3
        return vgg11(num_classes, in_channel)
    elif network == "resnet":
        return resnet(num_classes=num_classes)  # only support cifar, b/c in_channels
    elif network == "lenet":
        if dataset != "fashionmnist":
            raise ValueError("LeNet is only supported for input of size 28x28")
        return lenet()
    elif network == "lenet5":
        return lenet5(num_classes)
    elif network == "vit":
        return vit()
    else:
        raise ValueError("Network %s is not supported" % (network))


def load_data(dataset, num_train, batch_size, full_cifar=False):
    if dataset == "fashionmnist":
        return load_fmnist(num_train, batch_size)
    elif dataset == "cifar10":
        return load_cifar10(num_train, batch_size, shrink=not full_cifar)
    elif dataset == "imagenet":
        return load_imagenet(num_train, batch_size)
    else:
        raise ValueError("Dataset %s is not supported" % (dataset))


def set_seed(seed):
    logging.info(f"Set seed: {seed}")
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def accuracy(logits, targets, index=False):
    n = logits.shape[0]
    if targets.ndimension() == 2:
        _, y_trues = torch.max(targets, 1)
    else:
        y_trues = targets
    _, y_preds = torch.max(logits, 1)

    acc = (y_trues == y_preds).float().sum() * 100.0 / n

    if index:
        return acc, (y_trues == y_preds)
    else:
        return acc


def get_sharpness(net, criterion, dataloader, n_iters=10, tol=1e-2, verbose=False):
    v = eigen_hessian(
        net, criterion, dataloader, n_iters=n_iters, tol=tol, verbose=verbose
    )
    return v


def get_nonuniformity(net, criterion, dataloader, n_iters=10, tol=1e-2, verbose=False):
    v = eigen_variance(
        net, criterion, dataloader, n_iters=n_iters, tol=tol, verbose=verbose
    )
    return math.sqrt(v)


def eval_accuracy(model, criterion, dataloader, hard_sample=False):
    model.eval()
    n_batchs = len(dataloader)
    dataloader.idx = 0

    loss_t, acc_t = 0.0, 0.0
    hard_samples = []
    hard_targets = []
    with torch.no_grad():
        for _ in tqdm(range(n_batchs)):
            inputs, targets = next(iter(dataloader))
            inputs, targets = inputs.cuda(), targets.cuda()

            logits = model(inputs)
            acc, correct_index = accuracy(logits, targets, index=True)
            if hard_sample:
                if logits[~correct_index].shape[0] > 0:
                    loss_t += criterion(
                        logits[~correct_index], targets[~correct_index]
                    ).item()
            else:
                if targets.dim() == 1:
                    targets_one_hot = torch.zeros_like(logits)
                    targets = targets_one_hot.scatter_(1, targets.unsqueeze(1), 1)
                loss_t += criterion(logits, targets).item()
            acc_t += acc.item()
            hard_samples.append(inputs[~correct_index])
            hard_targets.append(targets[~correct_index])

    hard_samples = torch.cat(hard_samples, dim=0)
    hard_targets = torch.cat(hard_targets, dim=0)
    if hard_sample:
        hard_sample_loader = DataLoader(hard_samples, hard_targets, batch_size=1)
        return loss_t / n_batchs, acc_t / n_batchs, hard_sample_loader
    else:
        return (
            loss_t / n_batchs,
            acc_t / n_batchs,
            (
                DataLoader(dataloader.X, dataloader.y, batch_size=1)
                if hasattr(dataloader, "X")
                else torch.utils.data.DataLoader(dataloader.dataset, batch_size=1)
            ),
        )


def eval_output(model, dataloader):
    """
    batch_size is assumed to be 1
    """
    model.eval()
    n_batchs = len(dataloader)
    dataloader.idx = 0
    X = np.zeros((n_batchs))
    Y = np.zeros((n_batchs))
    for i in range(n_batchs):
        x, _ = next(dataloader)
        y = model(x)
        X[i] = x.item()
        Y[i] = y.item()
    return X, Y


"""
    get gradient with respect to W (sharpness in the case of mse loss)
    batch size of dataloader is assumed to be 1
    ndata: number of data points
    k: order of norm (the inequality in the paper is for k=1)
"""


def get_gradW(model, dataloader, ndata, k=1):
    assert dataloader.batch_size == 1
    gradtheta = 0
    gradW = 0
    B = 0
    for _ in range(ndata):
        X, _ = next(iter(dataloader))
        X = X.cuda()
        logits = model(X).reshape(1, -1)
        output_dim = logits.shape[1]
        normX = torch.linalg.vector_norm(X.flatten(), 2).item()
        old_gradW = gradW
        for j in range(output_dim):
            logit = logits[0][j]
            model.zero_grad()
            logit.backward(retain_graph=True)

            grad = [p.grad.detach().cpu().numpy() for p in model.parameters()]
            grad = [np.reshape(g, (-1)) for g in grad]
            cur_gradtheta = np.concatenate(grad)
            W = get_first_layer_weight(model)
            if W is not None:
                cur_gradW = W.grad.detach().cpu().numpy()
            else:
                cur_gradW = 0
            gradtheta += np.sum(cur_gradtheta**2)
            gradW += np.sum(cur_gradW**2)
        B += np.sqrt(gradW - old_gradW) / normX
    return np.sqrt(gradW / ndata), np.sqrt(gradtheta / ndata), B / ndata


def get_first_layer_weight(model):
    for layer in model.modules():
        if isinstance(layer, nn.Linear) or isinstance(layer, nn.Conv2d):
            return layer.weight
    return None


"""
Calculation of all metric in the feature space
"""


def get_dim(model, dataloader, ndata):
    assert dataloader.batch_size == 1
    dim = 0
    log_vol = 0
    G = 0
    A = 0
    for _ in range(ndata):
        X, y = next(iter(dataloader))
        X, y = X.cuda(), y.cuda()
        X.requires_grad = True
        logits = model(X).reshape(1, -1)
        grad_x = np.zeros((logits.shape[1], torch.numel(X)))
        old_G = G
        for j in range(logits.shape[1]):
            logit = logits[0][j]
            # model.zero_grad()
            grad = torch.autograd.grad(logit, X, retain_graph=True)[0]
            grad = grad.cpu().detach().numpy()
            grad = np.reshape(grad, (-1))
            grad_x[j, :] = grad
            G += np.sum(grad**2)
        sing_val = np.linalg.svd(grad_x, compute_uv=False)
        eig_val = sing_val**2
        A += np.max(sing_val)
        cur_dim = np.sum(eig_val) ** 2 / np.sum(eig_val**2)
        dim += cur_dim
        log_vol += cal_logvol(eig_val, cur_dim)
    return dim / ndata, log_vol / ndata, G / ndata, eig_val, A / ndata


def get_nmls(model, dataloader, ndata):
    nmls = 0
    harmonic = 0
    assert dataloader.batch_size == 1
    W_norm = None
    W0_norm = None
    norm_calculated = False
    for _ in range(ndata):
        activations = []
        weights = []
        output_numel = []
        handles = register(model, activations, weights, output_numel)
        X, y = next(iter(dataloader))
        X, y = X.cuda(), y.cuda()
        X.requires_grad = True
        logits = model(X).reshape(1, -1)
        unregister(handles)
        for i in range(len(activations)):
            grad_x = torch.zeros((logits.shape[1], activations[i].nelement()))
            for j in range(logits.shape[1]):
                logit = logits[0][j]
                # model.zero_grad()
                grad = torch.autograd.grad(logit, activations[i], retain_graph=True)[
                    0
                ].flatten()
                grad_x[j, :] = grad
            sing_val = torch.linalg.svdvals(grad_x)
            nmls += sing_val.max().item()
            eps = 0.0001
            if not isinstance(model, FNN):
                # for conv layers the two norm is not well defined, so using the frobenius norm instead
                # output numel is used to determine how many times the weights are repeated in a linear representation of the conv layer
                if len(weights[i].shape) == 2:
                    output_numel[i] = 1
                cur_norm = (
                    torch.linalg.matrix_norm(weights[i], "fro") ** 2
                ).sum() * output_numel[i]
                harmonic += cur_norm / (
                    torch.linalg.vector_norm(activations[i].flatten(), 2).item() ** 2
                    + eps
                )
            else:
                cur_norm = torch.linalg.matrix_norm(weights[i], 2) ** 2
                harmonic += cur_norm / (
                    torch.linalg.vector_norm(activations[i].flatten(), 2).item() ** 2
                    + eps
                )
            if not norm_calculated:
                if i == 0:
                    W0_norm = torch.sqrt(cur_norm)
                    W_norm = cur_norm
                else:
                    W_norm = W_norm + cur_norm
        norm_calculated = True
        assert harmonic.numel() == 1
    return nmls / ndata, harmonic / ndata, W0_norm, torch.sqrt(W_norm)


def get_hook(activations, weights, output_numel):
    def save_activations(module, input, output):
        if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
            input[0].retain_grad()
            activations.append(input[0])
            weights.append(module.weight)
            if output.dim() >= 2:
                output_numel.append(torch.tensor(output.shape[-2:]).prod().item())
            else:
                output_numel.append(output.numel())

    return save_activations


def register(model, activations, weights, output_numel):
    handles = []
    for layer in model.modules():
        if isinstance(layer, nn.Linear) or isinstance(layer, nn.Conv2d):
            handle = layer.register_forward_hook(
                get_hook(activations, weights, output_numel)
            )
            handles.append(handle)
    return handles


def unregister(handles):
    for handle in handles:
        handle.remove()


def cal_logvol(eig_val, dim):
    # return np.sum(np.log(eig_val[:math.floor(dim.item())])) / 2
    return np.sum(np.log(eig_val[:3])) / 2


def min_norm(dataloader, ndata):
    min_norm = 1e10
    for i in range(ndata):
        X, y = next(iter(dataloader))
        min_norm = min(min_norm, torch.linalg.matrix_norm(X, "fro").item())
    return min_norm


def quad_mean(dataloader, ndata):
    quad = 0
    for _ in range(ndata):
        X, _ = next(iter(dataloader))
        quad += 1 / torch.linalg.vector_norm(X.flatten(), 2).item() ** 2
    return np.sqrt(quad / ndata)


# adapted from https://github.com/kampmichael/RelativeFlatnessAndGeneralization/blob/0baa1f0c87db2860e1a4d8f675ff0347a8872b3f/CorrelationFlatnessGeneralization/utils.py
def calculateNeuronwiseHessians_fc_layer(model, dataloader, ndata, criterion):
    loss = torch.tensor(0.0).cuda()
    for _ in range(ndata):
        X, y = next(iter(dataloader))
        X, y = X.cuda(), y.cuda()
        logits = model(X)
        E = criterion(logits, y)
        loss += E
    loss /= ndata

    num_linear_layers = 0
    for p in model.modules():
        if isinstance(p, nn.Linear):
            num_linear_layers += 1
    idx = 0
    for p in model.modules():
        if not isinstance(p, nn.Linear):
            continue
        idx += 1
        if idx == num_linear_layers:
            feature_layer = p.weight
            break
    shape = feature_layer.shape

    layer_jacobian = torch.autograd.grad(
        loss, feature_layer, create_graph=True, retain_graph=True
    )
    drv2 = torch.empty(
        shape[1], shape[0], shape[0], shape[1], requires_grad=True
    ).cuda()
    for ind, n_grd in enumerate(layer_jacobian[0].T):
        for neuron_j in range(shape[0]):
            drv2[ind][neuron_j] = torch.autograd.grad(
                n_grd[neuron_j].cuda(), feature_layer, retain_graph=True
            )[0].cuda()
    # print("got hessian")

    trace_neuron_measure = 0.0
    maxeigen_neuron_measure = 0.0
    for neuron_i in range(shape[0]):
        neuron_i_weights = feature_layer[neuron_i, :].data.cpu().numpy()
        for neuron_j in range(shape[0]):
            neuron_j_weights = feature_layer[neuron_j, :].data.cpu().numpy()
            hessian = drv2[:, neuron_j, neuron_i, :]
            trace = np.trace(hessian.data.cpu().numpy())
            trace_neuron_measure += neuron_i_weights.dot(neuron_j_weights) * trace
            if neuron_j == neuron_i:
                eigenvalues = np.linalg.eigvalsh(hessian.data.cpu().numpy())
                maxeigen_neuron_measure += (
                    neuron_i_weights.dot(neuron_j_weights) * eigenvalues[-1]
                )

    return trace_neuron_measure, maxeigen_neuron_measure
