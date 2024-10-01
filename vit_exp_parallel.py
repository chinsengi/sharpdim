import copy
import json
import logging
import shutil
from time import time
import numpy as np
import timm
import os
import torch
import torch.nn as nn
from torch.autograd.functional import jacobian
import torch.nn.functional as F
from tqdm import tqdm
from src.data import load_imagenet
import argparse
import math
import torch.multiprocessing as mp


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
    parser.add_argument("--rho", type=float, default=0.1)
    parser.add_argument("--nonlinearity", type=str, default="sigmoid")
    parser.add_argument("--reduction", type=str, default="sum")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--skip_mls", action="store_true")
    parser.add_argument("--test_model", type=str, default="gcvit_small.in1k")
    parser.add_argument("--max_num_model", type=int, default=400)
    args = parser.parse_args()

    args.log = os.path.join(args.run, args.model, args.run_id)

    # args.device = use_gpu()
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


# Function to process each model independently on a given GPU
def process_model(model_name, device, args, nonlin, num_data, i):
    logging.warning(f"Processing model {i}-th model {model_name} on device {device}")
    try:
        model = timm.create_model(model_name, pretrained=True).to(device)
    except Exception as e:
        logging.warning(f"Error loading model {model_name}: {e}")
        return None
    model.eval()

    # Disable gradients for the model
    for p in model.parameters():
        p.requires_grad = False

    data_cfg = timm.data.resolve_data_config(model.pretrained_cfg)
    transform = timm.data.create_transform(**data_cfg)
    train_loader, _ = load_imagenet(50000, 1, transform)
    # MLS calculation (optional)
    if not args.skip_mls:
        logging.warning(f"Calculating MLS for model {model_name}")
        try:
            mls_avg, norm_mls_avg, mls_quality = eval_mls_adv(
                model, train_loader, args, nonlin, n_iters=num_data, device=device
            )
        except Exception as e:
            # to prevent out-of-memory error
            logging.warning(f"Error in calculating MLS for model {model_name}: {e}, skipping")
            return None
        if not mls_quality:
            logging.warning(f"MLS quality is not good for model {model_name}, skipping")
            return None
    else:
        mls_avg, norm_mls_avg = None, None

    # Sharpness calculation
    logging.warning(f"Calculating Sharpness for model {model_name}")
    try:
        sharpness, sharpness_quality = eval_average_sharpness(
            model,
            train_loader,
            torch.nn.MSELoss(reduction=args.reduction),
            n_iters=num_data,
            rho=args.rho,
            adaptive=not args.absolute_sharpness,
            nonlin=nonlin,
            device=device,
        )
    except Exception as e:
        # to prevent out-of-memory error
        logging.warning(f"Error in calculating sharpness for model {model_name}: {e}, skipping")
        return None

    if not sharpness_quality:
        logging.warning(
            f"Sharpness quality is not good for model {model_name}, skipping"
        )
        return None

    return mls_avg, norm_mls_avg, sharpness


# Main function to distribute the models across multiple GPUs
def main():
    args = get_args()
    logging.warning(f"Writing log file to {args.log}")
    config = json.dumps(vars(args), indent=2)
    logging.warning("===> Config:")
    logging.warning(config)
    logging.warning(f"Searching for models with name {args.model}")
    num_data = args.num_data
    devices = [
        f"cuda:{i}" for i in range(torch.cuda.device_count())
    ]  # List of available GPUs
    logging.warning(f"Devices: {devices}")
    model_list = timm.list_models(f"*{args.model}*", pretrained=True)

    start_time = time()

    # Create multiprocessing pool
    with mp.Pool(len(devices)) as pool:
        results = []

        # Distribute models to different GPUs
        for i, model_name in tqdm(enumerate(model_list)):
            if i >= args.max_num_model:
                break

            # Assign model to a GPU
            device = devices[i % len(devices)]
            args.device = device

            # Start parallel processes for each model
            if args.nonlinearity == "sigmoid":
                nonlin = F.sigmoid
            elif args.nonlinearity == "softmax":
                nonlin = nn.Softmax(dim=1)

            result = pool.apply_async(
                process_model,
                args=(
                    model_name,
                    device,
                    args,
                    nonlin,
                    num_data,
                    i,
                ),
            )
            results.append(result)


        # Collect results
        mls_list = []
        norm_mls_list = []
        sharpness_list = []

        for result in results:
            res = result.get()
            if res:
                mls_avg, norm_mls_avg, sharpness = res
                if mls_avg is not None:
                    mls_list.append(mls_avg)
                    norm_mls_list.append(norm_mls_avg)
                sharpness_list.append(sharpness)

            # Save the results
            save_npy(mls_list, args.log, "mls_list" + args.run_id)
            save_npy(sharpness_list, args.log, "sharpness_list" + args.run_id)
            save_npy(norm_mls_list, args.log, "norm_mls_list" + args.run_id)

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
    savefig(args.log, "norm_mls_vs_sharpness")
    plt.figure()
    plt.scatter(mls_list, sharpness_list)
    plt.xlabel("MLS")
    plt.ylabel("Adaptive Sharpness")
    # plt.title("MLS vs Sharpness")
    plt.annotate(
        f"Pearson correlation: {correlation:.2f}",
        xy=(0.05, 0.95),
        xycoords="axes fraction",
        fontsize=12,
        verticalalignment="top",
    )
    savefig(args.log, "mls_vs_sharpness")
    end_time = time()
    logging.warning(f"Time taken: {end_time - start_time} seconds")


if __name__ == "__main__":
    mp.set_start_method("spawn")  # Ensures proper GPU initialization for each process
    main()
