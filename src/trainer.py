import time
import numpy as np
import logging
import torch
from tqdm import tqdm
from .utils import get_dim, load_data, accuracy, get_gradW


def train(
    model,
    criterion,
    optimizer,
    dataloader,
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
    since = time.time()
    logging.info(n_iters)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, verbose=True)
    for iter_now in tqdm(range(n_iters)):
        optimizer.zero_grad()
        loss, acc = compute_minibatch_gradient(model, criterion, dataloader, batch_size)
        optimizer.step()

        if (iter_now+1) % 10 == 0:
            scheduler.step(loss)
            ntrain = 100
            dim_dataloader, _ = load_data(args.dataset, ntrain, batch_size=1)
            dim, log_vol = get_dim(model, dim_dataloader, args.dim_nsample)
            dim_list.append(dim)
            logvol_list.append(log_vol)
            sharpness_list.append(get_gradW(model, dim_dataloader, ntrain))
            acc_list.append(acc)
        # dim_list.append(get_dim(model, dim_dataloader))

        acc_avg = 0.9 * acc_avg + 0.1 * acc if acc_avg > 0 else acc
        loss_avg = 0.9 * loss_avg + 0.1 * loss if loss_avg > 0 else loss

        if iter_now % 200 == 0 and verbose:
            now = time.time()
            logging.info(
                "%d/%d, took %.0f seconds, train_loss: %.1e, train_acc: %.2f"
                % (iter_now + 1, n_iters, now - since, loss_avg, acc_avg)
            )
            since = time.time()
    return dim_list, sharpness_list, logvol_list, acc_list


def compute_minibatch_gradient(model, criterion, dataloader, batch_size):
    loss, acc = 0, 0
    n_loads = batch_size // dataloader.batch_size

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
