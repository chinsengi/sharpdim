import time
import numpy as np
import logging
import torch
from tqdm import tqdm
from .utils import (
    get_dim,
    load_data,
    accuracy,
    get_gradW,
    eval_accuracy,
    min_norm,
    quad_mean,
)
from .data import DataLoader


def train(
    model,
    criterion,
    optimizer,
    dataloader,
    test_loader,
    args,
    verbose=True,
):
    n_iters = args.n_iters
    batch_size = args.batch_size
    model.train()
    acc_avg, loss_avg = 0, 0

    dim_list = []
    sharpness_list = []
    logvol_list = []
    acc_list = []
    g_list = []
    eig_list = []
    loss_list = []
    test_acc_list = []
    test_loss_list = []
    W_list = []
    quad_list = [] # quadratic mean of the inputs
    gradW_list = [] # gradient of output w.r.t. the first layer weights
    A_list = [] # MLS
    B_list = [] # see appendix D.1

    since = time.time()

    # set up scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=10, verbose=True
    )

    dim_dataloader = DataLoader(dataloader.X, dataloader.y, batch_size=1)
    for iter_now in tqdm(range(n_iters), smoothing=0):
        optimizer.zero_grad()
        loss, acc = compute_minibatch_gradient(model, criterion, dataloader, batch_size)
        optimizer.step()

        acc_avg = 0.9 * acc_avg + 0.1 * acc if acc_avg > 0 else acc
        loss_avg = 0.9 * loss_avg + 0.1 * loss if loss_avg > 0 else loss

        if (iter_now + 1) % args.cal_freq == 0:
            if args.test_sample:
                test_loss, test_accuracy, dim_dataloader = eval_accuracy(
                    model, criterion, test_loader, hard_sample=args.hard_sample
                )
            else:
                test_loss, test_accuracy, _ = eval_accuracy(
                    model, criterion, test_loader
                )
            # test loss and accuracy
            test_acc_list.append(test_accuracy)
            test_loss_list.append(test_loss)

            # calculate dimension, log volume, G and eigenvalues.
            dim, log_vol, G, eig_val, A = get_dim(
                model, dim_dataloader, args.dim_nsample
            )
            dim_list.append(dim)
            logvol_list.append(log_vol)
            g_list.append(G)
            eig_list.append(eig_val)
            A_list.append(A)

            # calculate sharpness
            dim_dataloader.idx = dim_dataloader.idx - args.dim_nsample
            gradW, sharpness, B = get_gradW(model, dim_dataloader, args.dim_nsample)
            sharpness_list.append(sharpness)
            acc_list.append(acc_avg.item())
            loss_list.append(loss_avg)
            gradW_list.append(gradW)
            B_list.append(B)

            # calculate min input 2-norm and the norm of first layer.
            W_norm = None
            for param in model.parameters():
                if (
                    len(param.shape) == 2
                ):  # Check if the parameter is a weight matrix (2D)
                    W_norm = torch.linalg.matrix_norm(param, 2)
                break
            if W_norm is not None: W_list.append(W_norm.item())
            dim_dataloader.idx = dim_dataloader.idx - args.dim_nsample
            quad = quad_mean(dim_dataloader, args.dim_nsample) 
            quad_list.append(quad)

        if iter_now % 10000 == 0 and verbose:
            if args.use_scheduler:
                scheduler.step(loss)
            now = time.time()
            logging.info(
                "%d/%d, took %.0f seconds, train_loss: %.1e, train_acc: %.2f"
                % (iter_now + 1, n_iters, now - since, loss_avg, acc_avg)
            )
            since = time.time()

    '''
    make sure that the returned variables match
    save_list = [
            "dim_list",
            "sharpness_list",
            "logvol_list",
            "acc_list",
            "g_list",
            "eig_list",
            "loss_list",
            "test_acc_list",
            "test_loss_list",
            "W_list",
            "quad_list",
            "gradW_list",
            "A_list",
            "B_list",
        ]
    '''
    return (
        dim_list,
        sharpness_list,
        logvol_list,
        acc_list,
        g_list,
        eig_list,
        loss_list,
        test_acc_list,
        test_loss_list,
        W_list,
        quad_list,
        gradW_list,
        A_list,
        B_list,
    )


def compute_minibatch_gradient(model, criterion, dataloader, batch_size):
    loss, acc = 0, 0
    n_loads = batch_size // dataloader.batch_size
    assert n_loads == 1
    for i in range(n_loads):
        inputs, targets = next(dataloader)
        inputs, targets = inputs.cuda(), targets.cuda()

        logits = model(inputs)
        E = criterion(logits, targets)
        E.backward()

        loss += E.item()
        acc += accuracy(logits, targets)

    if n_loads > 1:
        for p in model.parameters():
            p.grad.data /= n_loads

    return loss / n_loads, acc / n_loads
