import logging
import math
import os
import time
from matplotlib import pyplot as plt
import numpy as np
import torch
import logging
import random
from .models.vgg import vgg11, vgg11_big
from .models.fnn import fnn
from .models.resnet import resnet
from .data import DataLoader, load_fmnist, load_cifar10
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
        logging.warning("weight file not found, training from scratch")


def save_model(model, optimizer, path, filename):
    create_dir(path)
    states = [model.state_dict(), optimizer.state_dict()]
    torch.save(states, os.path.join(path, filename))


def save_npy(obj, path, filename):
    create_dir(path)
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


# create directory
def create_dir(path="./model"):
    isExist = os.path.exists(path)
    if not isExist:
        os.makedirs(path)


def load_net(network, dataset, num_classes):
    if network == "fnn":
        return fnn(dataset, num_classes)
    elif network == "vgg":
        if dataset == "fashionmnist":
            in_channel = 1
        else:
            in_channel = 3
        return vgg11(num_classes, in_channel)
    elif network == "resnet":
        return resnet(num_classes=num_classes)  # only support cifar, b/c in_channels
    else:
        raise ValueError("Network %s is not supported" % (network))


def load_data(dataset, num_train, batch_size):
    if dataset == "fashionmnist":
        return load_fmnist(num_train, batch_size)
    elif dataset == "cifar10":
        return load_cifar10(num_train, batch_size)
    elif dataset == "1dfunction":
        raise NotImplementedError("Not implemented")
        # return load_1dfcn(train_per_class, batch_size)
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
    for i in range(n_batchs):
        inputs, targets = next(dataloader)
        inputs, targets = inputs.cuda(), targets.cuda()

        logits = model(inputs)
        acc, correct_index = accuracy(logits, targets, index=True)
        if hard_sample:
            if logits[~correct_index].shape[0] > 0:
                loss_t += criterion(logits[~correct_index], targets[~correct_index]).item()
        else:
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
        return loss_t / n_batchs, acc_t / n_batchs, DataLoader(dataloader.X, dataloader.y, batch_size=1)


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
    gradW = 0
    for _ in range(ndata):
        X, y = next(dataloader)
        X, y = X.cuda(), y.cuda()
        logits = model(X).reshape(1, -1)
        output_dim = logits.shape[1]
        for j in range(output_dim):
            logit = logits[0][j]
            model.zero_grad()
            logit.backward(retain_graph=True)

            grad = [p.grad.detach().cpu().numpy() for p in model.parameters()]
            grad = [np.reshape(g, (-1)) for g in grad]
            grad = np.concatenate(grad)
            gradW += np.sum(grad ** 2)
    return np.sqrt(gradW / ndata)


"""
    get gradient with respect to x
"""


def get_gradx(model, dataloader, ndata, k=1):
    assert dataloader.batch_size == 1
    gradx = 0
    for _ in range(ndata):
        X, y = next(dataloader)
        X, y = X.cuda(), y.cuda()
        X.requires_grad = True
        logits = model(X).reshape(1, -1)
        for j in range(logits.shape[1]):
            logit = logits[0][j]
            model.zero_grad()
            grad = torch.autograd.grad(logit, X, retain_graph=True)[0]
            grad = grad.detach().numpy()
            grad = np.reshape(grad, (-1))
            gradx += np.sum(grad ** (2 * k))
    return gradx / ndata


def get_dim(model, dataloader, ndata):
    assert dataloader.batch_size == 1
    dim = 0
    log_vol = 0
    G = 0
    for _ in range(ndata):
        X, y = next(dataloader)
        X, y = X.cuda(), y.cuda()
        X.requires_grad = True
        logits = model(X).reshape(1, -1)
        grad_x = np.zeros((logits.shape[1], torch.numel(X)))
        for j in range(logits.shape[1]):
            logit = logits[0][j]
            model.zero_grad()
            grad = torch.autograd.grad(logit, X, retain_graph=True)[0]
            grad = grad.cpu().detach().numpy()
            grad = np.reshape(grad, (-1))
            grad_x[j, :] = grad
            G += np.sum(grad ** 2)
        sing_val = np.linalg.svd(grad_x, compute_uv=False)
        eig_val = sing_val**2
        cur_dim = np.sum(eig_val) ** 2 / np.sum(eig_val**2)
        dim += cur_dim
        log_vol += cal_logvol(eig_val, cur_dim)
    return dim / ndata, log_vol / ndata, G / ndata, eig_val


def cal_logvol(eig_val, dim):
    # return np.sum(np.log(eig_val[:math.floor(dim.item())])) / 2
    return np.sum(np.log(eig_val[:3])) / 2
