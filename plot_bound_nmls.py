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


if __name__ == "__main__":
    # plt.style.use('science')
    dataset_list = ["fashionmnist"]
    # dataset_list = ["cifar10"]
    # dataset_list = ["fashionmnist", "cifar10"]
    for i, dataset in enumerate(dataset_list):
        run_id = 34
        with open(f"run/{dataset}/{run_id}/config.json") as f:
            config = json.load(f)
        data_list = [
            "nmls_list",
            "harm_list",
            "quad_list",
            "gradW_list",
            "sharpness_list",
            "test_acc_list",
        ]
        data = {}
        for list_name in data_list:
            data[list_name] = np.load(
                f"res/{dataset}/{run_id}/" + list_name + str(run_id) + ".npy"
            )
        list_len = len(data["nmls_list"])
        sample_freq = config["cal_freq"]
        breakpoint()
        df = pd.DataFrame(  
            {
                "NMLS": data["nmls_list"],
                "Bound": data["sharpness_list"] * np.sqrt(data["harm_list"]),
                "Sharpness": data["sharpness_list"],
                "Accuracy": data["test_acc_list"],
                "iteration (x5000)": np.arange(list_len) * sample_freq // 5000,
            }
        )
        df_plot = pd.melt(
            df,
            id_vars=["iteration (x5000)"],
            value_vars=['Bound', 'NMLS', 'Sharpness', 'Accuracy'],
            var_name="Bounds",
            value_name="Value",
        )
        df_plot = df_plot[df_plot["iteration (x5000)"] < 150]
        sns.set_theme(font_scale=2)
        sns.set_style("whitegrid")
        g = sns.relplot(
            data=df_plot,
            x="iteration (x5000)",
            y="Value",
            hue='Bounds',
            kind="line",
            palette="tab10",
            linewidth=1,
            height=6,
            aspect=1.4,
            dashes=False,
            ci=None
        )
        # g.set(yscale="log")
        sns.despine()
        savefig(
            "./image/bound", f"{dataset}_nmls_bound{run_id}", format="pdf", include_timestamp=True
        )
        # breakpoint()
