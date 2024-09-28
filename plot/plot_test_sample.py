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


def construct_df(run_ids, dataset, reconst=False, use_test_sample=False):
    if os.path.exists(f"res/df_{dataset}.pkl") and not reconst:
        dfs = pd.read_pickle(f"res/df_{dataset}.pkl")
    else:
        dfs = []
        for run_id in run_ids:
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
                "A_list"
            ]
            data = {}
            for list_name in data_list:
                data[list_name] = np.load(
                    f"res/{dataset}/{run_id}/" + list_name + str(run_id) + ".npy"
                )
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
                    "MLS": data["A_list"],
                    "Test samples": "Misclassified samples" if (run_id >= 34) else "All test samples",
                }
            )
            dfs.append(df)
        dfs = pd.concat(dfs)
    vars2idxs = ["run_id", "epoch", "network", "dataset", "iteration", "lr", "batch size", "Test samples"]
    vars2stack = [
        "Test loss",
        # "Test accuracy",
        "Sharpness",
        "Log volume",
        "MLS",
        "Log local dimension",
    ]
    df = dfs.set_index(vars2idxs)[vars2stack]
    df = (
        df.stack()
        .reset_index()
        .rename(columns={0: "value", "level_" + str(len(vars2idxs)): "variable"})
    )

    dfs.to_pickle(f"res/df_{dataset}.pkl")
    return df


if __name__ == "__main__":
    # plt.style.use('science')
    # dataset_list = ["fashionmnist"]
    dataset_list = ["cifar10"]
    # dataset_list = ["fashionmnist", "cifar10"]
    for i, dataset in enumerate(dataset_list):
        # run_ids = [i for i in range(1,11)]
        run_ids = (
            [i for i in range(12, 32)]
            if dataset == "fashionmnist"
            else [i for i in range(30, 38)]
        )
        print(f"run_ids to plot: {run_ids}")
        df = construct_df(run_ids, dataset, reconst=False)

        df1 = df[df["batch size"] == 20]
        # df1 = df1[df1["lr"] != 0.075]
        # df1['iteration_epoch'] = df1['iteration'] -  df1['epoch']*100
        df1['iteration (x1000)'] = df1['iteration'] //1000
        # df1 = df1[df1["iteration (x1000)"] <= 200]

        sns.set_theme(font_scale=1.5)
        sns.set_style("whitegrid")
        g = sns.relplot(
            data=df1,
            x="iteration (x1000)",
            y="value",
            hue="Test samples",
            col="variable",
            kind="line",
            facet_kws={"sharey": False},
            # palette="crest",
            # col_wrap=3,
            height=3,
            aspect=1.2,
            ci=None
            # errorbar="sd"
        )
        sns.despine()
        for item, ax in g.axes_dict.items():
            ax.grid(False)
            ax.set_title(item)
            if item =="Log local dimension":
                ax.set_ylim(0, .5)
            if item =='Test accuracy':
                if dataset == "fashionmnist":
                    ax.set_ylim(89, 91)
                else:
                    ax.set_ylim(93, 96)
            ax.spines['left'].set_color('black')
            ax.spines['left'].set_linewidth(1.0)  # Adjust the line width if needed
            ax.spines['bottom'].set_color('black')
            ax.spines['bottom'].set_linewidth(1.0) 
        ax = plt.gca()

        savefig(
            "./image", f"{dataset}_batch20", format="pdf", include_timestamp=True
        )