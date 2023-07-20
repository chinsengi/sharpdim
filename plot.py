import matplotlib.pyplot as plt
import numpy as np
from src.utils import create_dir
import os
import argparse


def plot_res(res_list, title, save_path, file_name):
    create_dir(save_path)
    plt.figure()
    epoch = np.arange(10, len(res_list) * 10+1, 10)
    plt.scatter(epoch, res_list, s=1)
    plt.title(title)
    plt.xlabel("iteration")
    # plt.ylabel('dimension')
    plt.savefig(os.path.join(save_path, file_name))
    # plt.close()


def get_args():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument("--run_id", type=str, default="0")
    args = argparser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    run_id = args.run_id
    plot_list = ["dim_list", "sharpness_list", "logvol_list", "acc_list"]
    title_list = ["dimensionality", "sharpness", "log volume", "accuracy"]
    for i in range(len(plot_list)):
        if os.path.isfile(f"res/{run_id}/" + plot_list[i] + run_id + ".npy"):
            cur_list = np.load(f"res/{run_id}/" + plot_list[i] + run_id + ".npy")
            plot_res(
                cur_list,
                title_list[i],
                f"./res/image/{run_id}",
                plot_list[i] + run_id + ".png",
            )
