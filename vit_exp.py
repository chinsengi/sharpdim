import copy
import json
import logging
import shutil
from time import time
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
from src.pert import eval_average_sharpness, eval_cov_sharpness
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
    parser.add_argument("--model", type=str, default="vit")
    parser.add_argument(
        "--verbose",
        type=str,
        default="info",
        help="Verbose level: info | debug | warning | critical",
    )
    parser.add_argument("--adaptive_sharpness", action="store_true")
    parser.add_argument("--num_data", type=int, default=1000)
    parser.add_argument("--rho", type=float, default=0.01)
    parser.add_argument("--nonlinearity", type=str, default="sigmoid")
    args = parser.parse_args()

    args.log = os.path.join(args.run, args.model, args.run_id)

    args.device = use_gpu()
    # specify logging configuration
    level = getattr(logging, args.verbose.upper(), None)
    if not isinstance(level, int):
        raise ValueError(f"level {args.verbose} not supported")

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


def main():
    # use timm to create and save the model
    # Ensure the model is saved to a file
    args = get_args()
    logging.info(f"Writing log file to {args.log}")
    config = json.dumps(vars(args), indent=2)
    logging.info("===> Config:")
    logging.info(config)
    logging.info(f"Searching for models with name {args.model}")
    model_list = timm.list_models(f"*{args.model}*", pretrained=True)
    logging.info(f"Number of models found: {len(model_list)}")
    mls_list = []
    sharpness_list = []
    norm_mls_list = []
    if args.nonlinearity == "sigmoid":
        nonlin = F.sigmoid
    elif args.nonlinearity == "softmax":
        nonlin = nn.Softmax(dim=1)
    for model_name in tqdm(model_list):
        # model_name = "gcvit_base.in1k"
        logging.info(f"processing model {model_name}")
        model = timm.create_model(
            model_name,
            pretrained=True,
        )
        if model.pretrained_cfg["num_classes"] != 1000:
            logging.info(
                f"Skipping model {model_name} as it is not trained on ImageNet"
            )
            continue
        model = model.to(args.device)
        model.eval()  # Set the model to evaluation mode

        # get the data transformation
        data_cfg = timm.data.resolve_data_config(model.pretrained_cfg)
        transform = timm.data.create_transform(**data_cfg)
        num_data = args.num_data
        train_loader, _ = load_imagenet(num_data, 1, transform)

        mls_sum = 0.0
        norm_mls_sum = 0.0
        with torch.no_grad():
            # calculate the mls
            for img, target in tqdm(train_loader):
                # logging.info(img.shape)
                # logging.info(target.shape)
                img, target = img.to(args.device), target.to(args.device)
                num_samples = 20  # Number of samples to use for jacobian computation
                alp = args.rho**2  # Noise variance
                # jacobian(model, img) # This takes 17 minutes to run
                noise = torch.randn(num_samples, *(img.shape[1:])).to(
                    args.device
                ) * math.sqrt(alp)
                out = model(img + noise)
                out = nonlin(out)
                cov = torch.cov(out.T) / alp
                assert cov.shape == (1000, 1000)
                mls = torch.linalg.matrix_norm(cov, ord=2).cpu().item()
                norm_mls = mls * torch.norm(img.flatten()).cpu().item()
                mls_sum = mls_sum + mls
                norm_mls_sum = norm_mls_sum + norm_mls

            mls_avg = mls_sum / num_data
            norm_mls_avg = norm_mls_sum / num_data
            logging.info(f"MLS for model {model_name}: {mls_avg}")
            logging.info(f"Normalized MLS for model {model_name}: {norm_mls_avg}")
            mls_list.append(mls_avg)
            norm_mls_list.append(norm_mls_avg)

            # calculate the sharpness
            # sharpness = eval_cov_sharpness(model, train_loader, rho=args.rho, n_iters=20)
            sharpness = (
                eval_average_sharpness(
                    model,
                    train_loader,
                    torch.nn.MSELoss(reduction="sum"),
                    n_iters=20,
                    rho=args.rho,
                    adaptive=args.adaptive_sharpness,
                    nonlin = nonlin
                )
                / args.rho**2
            )
            logging.info(f"Sharpness for model {model_name}: {sharpness}")
            sharpness_list.append(sharpness)

    lists = ["mls_list", "sharpness_list", "norm_mls_list"]
    for i in range(len(lists)):
        save_npy(
            eval(lists[i]),
            args.log,
            lists[i] + args.run_id,
        )
    correlation, _ = pearsonr(mls_list, sharpness_list)
    logging.info(f"Pearson correlation between MLS and Sharpness: {correlation}")
    norm_correlation, _ = pearsonr(norm_mls_list, sharpness_list)   
    logging.info(f"Pearson correlation between Normalized MLS and Sharpness: {norm_correlation}")
    plt.scatter(mls_list, sharpness_list)
    plt.xlabel("MLS")
    plt.ylabel("Sharpness")
    plt.title("MLS vs Sharpness")
    plt.annotate(
        f"Pearson correlation: {correlation:.2f}",
        xy=(0.05, 0.95),
        xycoords="axes fraction",
        fontsize=12,
        verticalalignment="top",
    )
    savefig(args.log)


if __name__ == "__main__":
    os.environ["HOME"] = "."
    main()
