import time
import numpy as np
import logging
import torch
from tqdm import tqdm

from .models.fnn import FNN
from .utils import (
    get_dim,
    accuracy,
    get_gradW,
    eval_accuracy,
    quad_mean,
    get_nmls,
    calculateNeuronwiseHessians_fc_layer
)
from .data import DataLoader
import torch.nn as nn
import logging

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
    W0_list = [] # norm of the first linear layer weights
    W_list = [] # norm of all linear weights
    quad_list = [] # quadratic mean of the inputs
    gradW_list = [] # gradient of output w.r.t. the first layer weights
    A_list = [] # MLS
    B_list = [] # see appendix D.1
    nmls_list = [] # NMLS
    harm_list = [] # harmonic mean of the inputs 
    rel_flatness_list = [] # relative flatness

    since = time.time()

    # set up scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=10
    )
    # breakpoint()
    if args.dataset == "imagenet":
        dim_dataloader = torch.utils.data.DataLoader(dataloader.dataset, batch_size=1)
    else:
        dim_dataloader = DataLoader(dataloader.X, dataloader.y, batch_size=1)
    for iter_now in tqdm(range(n_iters), smoothing=0):
        optimizer.zero_grad()
        loss, acc = compute_minibatch_gradient(model, criterion, dataloader, args)
        optimizer.step()

        acc_avg = 0.9 * acc_avg + 0.1 * acc if acc_avg > 0 else acc
        loss_avg = 0.9 * loss_avg + 0.1 * loss if loss_avg > 0 else loss

        if (iter_now + 1) % args.cal_freq == 0:
            logging.info("start accuray calculation")
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

            # calculation NMLS
            logging.info("start NMLS calculation")
            nmls, harmonic, W0_norm, W_norm = get_nmls(model, dim_dataloader, args.dim_nsample)
            nmls_list.append(nmls)
            harm_list.append(harmonic.item())
            W_list.append(W_norm.item())
            W0_list.append(W0_norm.item())

            # calculate dimension, log volume, G and eigenvalues.
            logging.info("start dimension calculation")
            dim_dataloader.idx = dim_dataloader.idx - args.dim_nsample
            dim, log_vol, G, eig_val, A = get_dim(
                model, dim_dataloader, args.dim_nsample
            )
            dim_list.append(dim)
            logvol_list.append(log_vol)
            g_list.append(G)
            eig_list.append(eig_val)
            A_list.append(A)

            # calculate sharpness
            logging.info("start sharpness calculation")
            dim_dataloader.idx = dim_dataloader.idx - args.dim_nsample
            gradW, sharpness, B = get_gradW(model, dim_dataloader, args.dim_nsample)
            sharpness_list.append(sharpness)
            acc_list.append(acc_avg.item())
            loss_list.append(loss_avg)
            gradW_list.append(gradW)
            B_list.append(B)

            dim_dataloader.idx = dim_dataloader.idx - args.dim_nsample
            quad = quad_mean(dim_dataloader, args.dim_nsample) 
            quad_list.append(quad)

            # calculate the relative flatness
            logging.info("start relative flatness calculation")
            trace_nm, maxeigen_nm = calculateNeuronwiseHessians_fc_layer(model, dim_dataloader, args.dim_nsample, criterion)
            rel_flatness_list.append(trace_nm)


        if (iter_now+1) % 10000 == 0 and verbose:
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
            "nmls_list",
            "harm_list",
            "rel_flatness_list",
            "W0_list",
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
        nmls_list,
        harm_list,
        rel_flatness_list,
        W0_list,
    )

def compute_minibatch_gradient(model, criterion, dataloader, args):
    loss, acc = 0, 0
    # breakpoint()
    inputs, targets = next(iter(dataloader))
    inputs, targets = inputs.cuda(), targets.cuda()

    logits = model(inputs)

    if targets.dim() == 1:
        targets_one_hot = torch.zeros_like(logits)
        targets = targets_one_hot.scatter_(1, targets.unsqueeze(1), 1)
    E = criterion(logits, targets)
    E.backward()

    loss += E.item()
    acc += accuracy(logits, targets)

    if args.use_layer_norm:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)

    return loss, acc
