import time
import numpy as np
from .utils import get_dim, load_data, accuracy


def train(
    model,
    criterion,
    optimizer,
    dataloader,
    args,
    batch_size,
    n_iters=50000,
    verbose=True,
):
    model.train()
    acc_avg, loss_avg = 0, 0

    dim_list = []
    dim_dataloader, _ = load_data(args.dataset, args.dim_nsample, batch_size=1)
    since = time.time()
    for iter_now in range(n_iters):
        optimizer.zero_grad()
        loss, acc = compute_minibatch_gradient(model, criterion, dataloader, batch_size)
        optimizer.step()

        dim_list.append(get_dim(model.features, dim_dataloader))
        # dim_list.append(get_dim(model, dim_dataloader))

        acc_avg = 0.9 * acc_avg + 0.1 * acc if acc_avg > 0 else acc
        loss_avg = 0.9 * loss_avg + 0.1 * loss if loss_avg > 0 else loss

        if iter_now % 200 == 0 and verbose:
            now = time.time()
            print(
                "%d/%d, took %.0f seconds, train_loss: %.1e, train_acc: %.2f"
                % (iter_now + 1, n_iters, now - since, loss_avg, acc_avg)
            )
            since = time.time()
    return dim_list


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
        acc += accuracy(logits.data, targets)

    for p in model.parameters():
        p.grad.data /= n_loads

    return loss / n_loads, acc / n_loads
