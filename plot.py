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
import math


def plot_res(res_list, title, save_path, file_name):
    create_dir(save_path)
    plt.figure()
    res_list = res_list[: (len(res_list) // 20) * 20].reshape(20, -1).mean(axis=0)
    # epoch = np.arange(10, len(res_list) * 10 + 1, 10)
    # plt.scatter(epoch, res_list, s=1)
    plt.plot(res_list)
    plt.title(title)
    plt.xlabel("iteration")
    # plt.ylabel('dimension')
    plt.savefig(os.path.join(save_path, file_name))
    # plt.close()


def construct_df(run_ids, dataset, recompute=False):
    if os.path.exists(f"res/df_{dataset}.pkl") and not recompute:
        dfs = pd.read_pickle(f"res/df_{dataset}.pkl")
    else:
        dfs = []
        for run_id in run_ids:
            if run_id == 5:
                continue
            # load the json file
            with open(f"run/{dataset}/{run_id}/config.json") as f:
                config = json.load(f)
            data_list = [
                "dim_list",
                "sharpness_list",
                "logvol_list",
                "acc_list",
                "g_list",
                "eig_list",
                "loss_list",
                "test_acc_list",
                "test_loss_list",
            ]
            data = {}
            for list_name in data_list:
                data[list_name] = np.load(
                    f"res/{run_id}/" + list_name + str(run_id) + ".npy"
                )
            # if config['lr']==0.1:
            #     breakpoint()
            train_size = 60000 if config["dataset"] == "fashionmnist" else 10000
            batch_size = config["batch_size"]
            n_iter_per_epoch = math.ceil(train_size / batch_size)
            list_len = len(data["dim_list"])
            print(
                f"batch_size: {config['batch_size']}, lr: {config['lr']}, list_len: {list_len}"
            )
            sample_freq = config["cal_freq"] 
            df = pd.DataFrame(
                {
                    "run_id": run_id,
                    "dataset": config["dataset"],
                    "network": config["network"],
                    "lr": config["lr"],
                    "batch size": config["batch_size"],
                    "n_iters": config["n_iters"],
                    "epoch": (np.arange(list_len) * sample_freq)// (n_iter_per_epoch),
                    "iteration": np.arange(list_len) * sample_freq,
                    "Log local dimension": np.log(data["dim_list"]),
                    "Sharpness": data["sharpness_list"],
                    "Log volume": data["logvol_list"],
                    "Train accuracy": data["acc_list"],
                    "Train loss": data["loss_list"],
                    "Test accuracy": data["test_acc_list"],
                    "Test loss": data["test_loss_list"],
                    "G": data["g_list"],
                    "eig": data["eig_list"].tolist(),
                }
            )
            dfs.append(df)
        dfs = pd.concat(dfs)
    vars2idxs = ["run_id", "epoch", "network", "dataset", "iteration", "lr", "batch size"]
    vars2stack = [
        "Train loss",
        # "Test loss",
        "Test accuracy",
        "Sharpness",
        "Log volume",
        "G",
        "Log local dimension",
    ]
    df = dfs.set_index(vars2idxs)[vars2stack]
    df = (
        df.stack()
        .reset_index()
        .rename(columns={0: "value", "level_" + str(len(vars2idxs)): "variable"})
    )
    # breakpoint()
    dfs.to_pickle(f"res/df_{dataset}.pkl")
    return df


if __name__ == "__main__":
    # plt.style.use('science')
    # dataset_list = ["fashionmnist"]
    dataset_list = ["cifar10"]
    # dataset_list = ["fashionmnist", "cifar10"]
    for i, dataset in enumerate(dataset_list):
        # run_ids = [i for i in range(0, 20)] + [i for i in range(53, 73)] if dataset == "cifar10" else range(21, 26)
        run_ids = [i for i in range(11,21)]
        print(f"run_ids to plot: {run_ids}")
        df = construct_df(run_ids, dataset, recompute=True)
        # breakpoint()
        df1 = df[df["batch size"] == 20]
        # df1 = df1[df1["lr"] != 0.075]
        # df1['iteration_epoch'] = df1['iteration'] -  df1['epoch']*100
        df1['iteration (x1000)'] = df1['iteration'] //1000
        # df1 = df1[df1["iteration (x1000)"] <= 200]
        # breakpoint()
        sns.set(font_scale=3)
        sns.set_style("whitegrid")
        g = sns.relplot(
            data=df1,
            x="iteration (x1000)",
            y="value",
            hue="lr",
            col="variable",
            kind="line",
            facet_kws={"sharey": False},
            palette="crest",
            col_wrap=3,
            height=6,
            aspect=1.4
            # errorbar="sd"
        )
        sns.despine()
        for item, ax in g.axes_dict.items():
            ax.grid(False)
            ax.set_title(item)
            # if item =="Log local dimension":
            #     ax.set_ylim(0, .5)
            # if item =='Test accuracy':
            #     ax.set_ylim(85, 91)
            ax.spines['left'].set_color('black')
            ax.spines['left'].set_linewidth(1.0)  # Adjust the line width if needed
            ax.spines['bottom'].set_color('black')
            ax.spines['bottom'].set_linewidth(1.0) 
        ax = plt.gca()

        savefig(
            "./image", f"{dataset}_batch20", format="pdf", include_timestamp=True
        )

        df2 = df[df["lr"] == 0.1]
        # breakpoint()
        df2['iteration (x1000)'] = df2['iteration'] //1000
        # df2 = df2[df2["iteration (x1000)"] <= 100]
        g = sns.relplot(
            data=df2,
            x="iteration (x1000)",
            # x="epoch",
            y="value",
            hue="batch size",
            col="variable",
            kind="line",
            facet_kws={"sharey": False},
            palette="Set1",
            col_wrap=3,
            height=6,
            aspect=1.4
            # errorbar="sd"
        )
        sns.despine()
        for item, ax in g.axes_dict.items():
            ax.grid(False)
            ax.set_title(item)
            # if item =="Log local dimension":
            #     ax.set_ylim(0, .5)
            # if item =='Test accuracy':
            #     ax.set_ylim(85, 91)
            ax.spines['left'].set_color('black')
            ax.spines['left'].set_linewidth(1.0)  # Adjust the line width if needed
            ax.spines['bottom'].set_color('black')
            ax.spines['bottom'].set_linewidth(1.0) 
        savefig(
            "./image", f"{dataset}_lr01", format="pdf", include_timestamp=True
        )

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
