import time
import numpy as np
import logging
import torch
from tqdm import tqdm
from .utils import get_dim, load_data, accuracy, get_gradW, eval_accuracy
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
                test_loss, test_accuracy, dim_dataloader = eval_accuracy(model, criterion, test_loader, hard_sample=args.hard_sample)
            else:
                test_loss, test_accuracy, _ = eval_accuracy(model, criterion, test_loader)
            # test loss and accuracy
            test_acc_list.append(test_accuracy)
            test_loss_list.append(test_loss)

            # calculate dimension, log volume, G and eigenvalues. 
            dim, log_vol, G, eig_val = get_dim(model, dim_dataloader, args.dim_nsample)
            dim_list.append(dim)
            logvol_list.append(log_vol)
            g_list.append(G)
            eig_list.append(eig_val)

            # calculate sharpness
            sharpness_list.append(get_gradW(model, dim_dataloader, args.dim_nsample))
            acc_list.append(acc_avg.item())
            loss_list.append(loss_avg)


        if iter_now % 1000 == 0 and verbose:
            if args.use_scheduler:
                scheduler.step(loss)
            now = time.time()
            logging.info(
                "%d/%d, took %.0f seconds, train_loss: %.1e, train_acc: %.2f"
                % (iter_now + 1, n_iters, now - since, loss_avg, acc_avg)
            )
            since = time.time()
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
    )


def compute_minibatch_gradient(model, criterion, dataloader, batch_size):
    loss, acc = 0, 0
    n_loads = batch_size // dataloader.batch_size
    assert(n_loads==1)
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
