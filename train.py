import os
import argparse
import json
import traceback
import torch
import logging
import shutil
from src.utils import set_seed
import time
import random
from src.trainer import train
from src.utils import (
    load_net,
    load_data,
    eval_accuracy,
    savefig,
    use_gpu,
    save_npy,
    save_model,
)


def get_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run", type=str, default="run", help="Path for saving running related data."
    )
    parser.add_argument(
        "--run_id", type=str, default="0", help="id used to identify different runs"
    )
    parser.add_argument(
        "--verbose",
        type=str,
        default="info",
        help="Verbose level: info | debug | warning | critical",
    )
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument(
        "--random", action="store_true", help="whether to set random seed"
    )
    parser.add_argument(
        "--dataset",
        default="fashionmnist",
        help="dataset, [fashionmnist] | cifar10, imagenet",
    )
    parser.add_argument("--network", default="fnn", help="network, [fnn] | vgg, resnet")
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--optimizer", default="sgd", help="optimizer, [sgd]")
    parser.add_argument(
        "--n_iters",
        type=int,
        default=150000,
        help="number of iteration used to train nets, [150000]",
    )
    parser.add_argument("--batch_size", type=int, default=8, help="batch size, [8]")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument("--momentum", type=float, default="0.0", help="momentum, [0.0]")
    parser.add_argument("--train_size", type=int, default=60000)
    parser.add_argument(
        "--dim_nsample",
        type=int,
        default=20,
        help="number of samples to compute dimension",
    )
    parser.add_argument("--comment", type=str, default="")
    parser.add_argument(
        "--loss", type=str, default="mse", help="loss function, [mse] | cross_entropy"
    )
    parser.add_argument(
        "--use_scheduler",
        action="store_true",
        help="whether to use learning rate scheduler",
    )
    parser.add_argument(
        "--cal_freq",
        type=int,
        default=100,
        help="how many data points within per epoch",
    )
    parser.add_argument(
        "--hard_sample",
        action="store_true",
        help="whether to use hard samples to compute dimension",
    )
    parser.add_argument(
        "--test_sample",
        action="store_true",
        help="whether to use test samples to compute dimension",
    )
    parser.add_argument(
        "--use_layer_norm", action="store_true", help="whether to use layer norm in fnn"
    )
    parser.add_argument(
        "--nonlinearity", default="tanh", help="nonlinearity of FFN, [tanh] | relu"
    )
    parser.add_argument(
        "--full_cifar",
        action="store_true",
        help="whether to shrink the cifar10 dataset and use only 2 classes, help with achieving 0 training error.",
    )
    args = parser.parse_args()

    args.log = os.path.join(args.run, args.dataset, args.run_id)

    # use the same args
    # with open(f"./run/{args.run_id}/config.json") as f:
    #     args = json.load(f)
    #     args = argparse.Namespace(**args)

    args.device = use_gpu()
    # specify logging configuration
    level = getattr(logging, args.verbose.upper(), None)
    if not isinstance(level, int):
        raise ValueError(f"level {args.verbose} not supported")

    if os.path.exists(args.log):
        shutil.rmtree(args.log)
    if not os.path.exists(args.log):
        os.makedirs(args.log)

    handler1 = logging.StreamHandler()
    handler2 = logging.FileHandler(os.path.join(args.log, "stdout.txt"))
    formatter = logging.Formatter(
        "%(levelname)s - %(filename)s - %(asctime)s - %(message)s"
    )
    handler1.setFormatter(formatter)
    handler2.setFormatter(formatter)
    logger = logging.getLogger()
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    logger.setLevel(level)

    return args


def get_optimizer(net, args):
    if args.optimizer == "sgd":
        return torch.optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum)
    elif args.optimizer == "adam":
        return torch.optim.Adam(net.parameters(), lr=args.lr)
    else:
        raise ValueError("optimizer %s has not been supported" % (args.optimizer))


def main():
    args = get_args()

    # args.random = False
    if not args.random:
        set_seed(args.seed)
    else:
        args.seed = random.randint(0, 10000)
        set_seed(args.seed)
    print(f"Writing log file to {args.log}")
    logging.info(f"Exp instance id = {os.getpid()}")
    logging.info(f"Exp comment = {args.comment}")

    if args.loss == "mse":
        criterion = torch.nn.MSELoss()
    elif args.loss == "cross_entropy":
        criterion = torch.nn.CrossEntropyLoss()

    if args.network == "vit":
        args.dataset = "imagenet"
    train_loader, test_loader = load_data(
        args.dataset, args.train_size, batch_size=args.batch_size, full_cifar=args.full_cifar
    )
    # args.n_iters = args.n_epochs * len(train_loader)

    # write config
    config = json.dumps(vars(args), indent=2)
    logging.info("===> Config:")
    logging.info(config)
    with open(os.path.join(args.log, "config.json"), "w") as f:
        f.write(config)

    net = load_net(
        args.network,
        args.dataset,
        args.num_classes,
        args.nonlinearity,
        args.use_layer_norm,
    )
    net = net.to(args.device)
    optimizer = get_optimizer(net, args)
    logging.info(optimizer)

    logging.info("===> Architecture:")
    logging.info(net)

    logging.info("===> Start training")
    try:
        # torch.backends.cuda.preferred_linalg_library(backend="magma")
        lists = train(
            net,
            criterion,
            optimizer,
            train_loader,
            test_loader,
            args,
            verbose=True,
        )
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
        for i in range(len(lists)):
            save_npy(
                lists[i],
                f"res/{args.dataset}/{args.run_id}",
                save_list[i] + args.run_id,
            )

        train_loss, train_accuracy, _ = eval_accuracy(net, criterion, train_loader)
        test_loss, test_accuracy, _ = eval_accuracy(net, criterion, test_loader)
        logging.info("===> Solution: ")
        logging.info("\t train loss: %.2e, acc: %.2f" % (train_loss, train_accuracy))
        logging.info("\t test loss: %.2e, acc: %.2f" % (test_loss, test_accuracy))
    except:
        logging.error(traceback.format_exc())
    save_model(net, optimizer, "res/models/", args.dataset + args.run_id + ".pkl")


if __name__ == "__main__":
    main()
