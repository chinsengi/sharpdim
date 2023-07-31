import matplotlib.pyplot as plt
import numpy as np
from src.utils import create_dir, savefig
import os
import argparse
import pandas as pd
import seaborn as sns
import json
from scipy.signal import savgol_filter
import scienceplots as sp


def plot_res(res_list, title, save_path, file_name):
    create_dir(save_path)
    plt.figure()
    res_list = res_list[:(len(res_list)//20)*20].reshape(20,-1).mean(axis=0)
    # epoch = np.arange(10, len(res_list) * 10 + 1, 10)
    # plt.scatter(epoch, res_list, s=1)
    plt.plot(res_list)
    plt.title(title)
    plt.xlabel("iteration")
    # plt.ylabel('dimension')
    plt.savefig(os.path.join(save_path, file_name))
    # plt.close()


def construct_df(run_ids, dataset, recompute=False):
    if os.path.exists(f"res/df_{dataset}.csv") and not recompute:
        return pd.read_csv(f"res/df_{dataset}.csv")
    dfs = []
    for run_id in run_ids:
        # load the json file
        with open(f"run/{run_id}/config.json") as f:
            config = json.load(f)
        data_list = [
            "dim_list",
            "sharpness_list",
            "logvol_list",
            "acc_list",
            "g_list",
            # "eig_list",
            "loss_list",
        ]
        data = {}
        for list_name in data_list:
            data[list_name] = np.load(
                f"res/{run_id}/" + list_name + str(run_id) + ".npy"
            )
            # if list_name != "eig_list":
            #     data[list_name] = savgol_filter(data[list_name], 51, 3)
        train_size = 30000 if config["dataset"] == "fashionmnist" else 10000
        batch_size = config["batch_size"]
        n_iter_per_epoch = train_size // batch_size
        list_len = len(data["dim_list"])
        sample_freq = config["n_iters"] / list_len
        # import pdb; pdb.set_trace()
        df = pd.DataFrame(
            {
                "run_id": run_id,
                "dataset": config["dataset"],
                "network": config["network"],
                "lr": config["lr"],
                "batch_size": config["batch_size"],
                "n_iters": config["n_iters"],
                "epoch": np.arange(list_len) // (n_iter_per_epoch / sample_freq),
                "iteration": np.arange(list_len),
                "dimension": data["dim_list"],
                "sharpness": data["sharpness_list"],
                "log_volume": data["logvol_list"],
                "train_accuracy": data["acc_list"],
                "train_loss": data["loss_list"],
                "G": data["g_list"],
            }
        )
        dfs.append(df)
    dfs = pd.concat(dfs)
    dfs.to_csv(f"res/df_{dataset}.csv")
    return dfs


def get_args():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument("--run_id", type=str, default="0")
    argparser.add_argument("--dataset", type=str, default="[fashionmnist] | cifar10")
    args = argparser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    run_id = args.run_id
    # plt.style.use('science')
    fig, axes = plt.subplots(2, 4, figsize=(80, 40))
    dataset_list = ["cifar10"]
    # dataset_list = ["fashionmnist", "cifar10"]
    plot_var_list = ["sharpness", "log_volume","dimension", "G"]
    for i, dataset in enumerate(dataset_list):
        run_ids = range(0, 24) if dataset == "cifar10" else range(21, 26)
        df = construct_df(run_ids, dataset, recompute=False)
        # import pdb; pdb.set_trace()
        for j, var in enumerate(plot_var_list):
            ax = axes[i*2, j]
            # sns.set_theme(style="darkgrid")
            sns.lineplot(
                data=df[df["batch_size"] == 20],
                x="epoch",
                y=var,
                hue="lr",
                err_style="band",
                ax=ax
            )
        for j, var in enumerate(plot_var_list):
            ax = axes[i*2+1, j]
            # sns.set_theme(style="darkgrid")
            sns.lineplot(
                data=df[df["lr"] == .1],
                x="epoch",
                y=var,
                hue="batch_size",
                err_style="band",
                estimator=np.median,
                ax=ax
            )
    savefig('./res/image', "all", format='png')


    # plot_list = ["dim_list", "sharpness_list", "logvol_list", "acc_list"]
    # title_list = ["dimensionality", "sharpness", "log volume", "accuracy"]
    # for i in range(len(plot_list)):
    #     if os.path.isfile(f"res/{run_id}/" + plot_list[i] + run_id + ".npy"):
    #         cur_list = np.load(f"res/{run_id}/" + plot_list[i] + run_id + ".npy")
    #         plot_res(
    #             cur_list,
    #             title_list[i],
    #             f"./res/image/{run_id}",
    #             plot_list[i] + run_id + ".png",
    #         )
