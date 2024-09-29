import copy
import json
import logging
import shutil
from time import time
import numpy as np
import timm
import logging
import os
import torch
import torch.nn as nn
from torch.autograd.functional import jacobian
import torch.nn.functional as F
from tqdm import tqdm
from src.data import load_imagenet
import argparse
import math

from src.utils import save_npy, savefig, use_gpu
from src.pert import eval_average_sharpness, eval_cov_sharpness, eval_mls, eval_mls_adv
from matplotlib import pyplot as plt
from scipy.stats import pearsonr


def get_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run", type=str, default="run", help="Path for saving running related data."
    )
    parser.add_argument(
        "--run_id", type=str, default="0", help="id used to identify different runs"
    )
    parser.add_argument("--model", type=str, default="vit_small")
    parser.add_argument(
        "--verbose",
        type=str,
        default="info",
        help="Verbose level: info | debug | warning | critical",
    )
    parser.add_argument("--absolute_sharpness", action="store_true")
    parser.add_argument(
        "--num_data",
        type=int,
        default=100,
        help="Number of data points to use for sharpness approximation",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=256,
        help="Number of samples to use for jacobian computation",
    )
    parser.add_argument("--rho", type=float, default=0.01)
    parser.add_argument("--nonlinearity", type=str, default="sigmoid")
    parser.add_argument("--reduction", type=str, default="sum")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--skip_mls", action="store_true")
    parser.add_argument("--test_model", type=str, default="gcvit_small.in1k")
    args = parser.parse_args()

    args.log = os.path.join(args.run, args.model, args.run_id)

    args.device = use_gpu()
    # specify logging configuration
    level = getattr(logging, args.verbose.upper(), None)
    if not isinstance(level, int):
        raise ValueError(f"level {args.verbose} not supported")

    if args.run_id != "0":
        assert not os.path.exists(args.log), f"{args.log} already exists"
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


def check_model_input(model):
    input_size = model.pretrained_cfg["input_size"][-1]
    return input_size <= 384


def main():
    # use timm to create and save the model
    # Ensure the model is saved to a file
    args = get_args()
    logging.warning(f"Writing log file to {args.log}")
    config = json.dumps(vars(args), indent=2)
    logging.warning("===> Config:")
    logging.warning(config)
    logging.warning(f"Searching for models with name {args.model}")
    model_list = timm.list_models(f"*{args.model}*", pretrained=True)
    logging.warning(f"Number of models found: {len(model_list)}")
    mls_list = []
    sharpness_list = []
    norm_mls_list = []
    if args.nonlinearity == "sigmoid":
        nonlin = F.sigmoid
    elif args.nonlinearity == "softmax":
        nonlin = nn.Softmax(dim=1)
    testing = args.test
    for i, model_name in tqdm(enumerate(model_list)):
        logging.warning(f"Processing {i}/{len(model_list)}-th model: {model_name}")
        if testing:
            model_name = args.test_model
            logging.warning(f"Testing model {model_name}")
        logging.warning(f"processing model {model_name}")
        model = timm.create_model(
            model_name,
            pretrained=True,
        )
        if not check_model_input(model):
            logging.warning(
                f"Skipping model {model_name} as its input exceeds size 384*384"
            )
            continue
        if model.pretrained_cfg["num_classes"] != 1000:
            logging.warning(
                f"Skipping model {model_name} as it is not trained on ImageNet"
            )
            continue
        model = model.to(args.device)
        model.eval()  # Set the model to evaluation mode
        for p in model.parameters():
            p.requires_grad = False
        # get the data transformation
        data_cfg = timm.data.resolve_data_config(model.pretrained_cfg)
        transform = timm.data.create_transform(**data_cfg)
        num_data = args.num_data
        train_loader, _ = load_imagenet(50000, 1, transform)

        logging.warning(f"Calculating MLS for model {model_name}")
        if not args.skip_mls:
            mls_avg, norm_mls_avg = eval_mls_adv(
                model, train_loader, args, nonlin, n_iters=num_data
            )
            mls_list.append(mls_avg)
            norm_mls_list.append(norm_mls_avg)
        else:
            logging.warning(f"Skipping MLS calculation for model {model_name}")
        
        logging.warning(f"Calculating Sharpness for model {model_name}")
        with torch.no_grad():
            # calculate the mls
            # mls_avg, norm_mls_avg = eval_mls(
            #     model,
            #     train_loader,
            #     args,
            #     nonlin,
            #     model_name,
            #     n_iters=num_data,
            #     num_samples=args.num_samples,
            # )

            # calculate the sharpness
            # sharpness = eval_cov_sharpness(model, train_loader, rho=args.rho, n_iters=20)
            sharpness = (
                eval_average_sharpness(
                    model,
                    train_loader,
                    torch.nn.MSELoss(reduction=args.reduction),
                    n_iters=num_data,
                    rho=args.rho,
                    adaptive=not args.absolute_sharpness,
                    nonlin=nonlin,
                )
            )
            logging.warning(f"Sharpness for model {model_name}: {sharpness}")
            sharpness_list.append(sharpness)
        if testing:
            assert False

    lists = ["mls_list", "sharpness_list", "norm_mls_list"]
    for i in range(len(lists)):
        save_npy(
            eval(lists[i]),
            args.log,
            lists[i] + args.run_id,
        )
    correlation, _ = pearsonr(mls_list, sharpness_list)
    logging.warning(f"Pearson correlation between MLS and Sharpness: {correlation}")
    norm_correlation, _ = pearsonr(norm_mls_list, sharpness_list)
    logging.warning(
        f"Pearson correlation between Normalized MLS and Sharpness: {norm_correlation}"
    )
    plt.scatter(norm_mls_list, sharpness_list)
    plt.xlabel("Normalized MLS")
    plt.ylabel("Adaptive Sharpness")
    # plt.title("MLS vs Sharpness")
    plt.annotate(
        f"Pearson correlation: {norm_correlation:.2f}",
        xy=(0.05, 0.95),
        xycoords="axes fraction",
        fontsize=12,
        verticalalignment="top",
    )
    savefig(args.log)


if __name__ == "__main__":
    os.environ["HOME"] = "."
    main()
