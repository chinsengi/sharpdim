import matplotlib.pyplot as plt
import numpy as np
from src.utils import create_dir, savefig
import pandas as pd
import seaborn as sns
import json
from scipy.signal import savgol_filter
from scipy.stats import pearsonr


def smooth(x, window_size, polyorder=3):
    return savgol_filter(x, window_size, polyorder)


def corrfunc(x, y, hue=None, ax=None, **kws):
    """Plot the correlation coefficient in the top left hand corner of a plot."""
    r, _ = pearsonr(x, y)
    ax = ax or plt.gca()
    ax.annotate(f"ρ = {r:.2f}", xy=(0.1, 0.9), xycoords=ax.transAxes)


if __name__ == "__main__":
    dataset = "fashionmnist"
    dataset = "cifar10"
    start_id = 400
    end_id = 500
    run_ids = [i for i in range(start_id, end_id)]
    stat = {}
    data_list = [
        "dim",
        "sharpness",
        "logvol",
        "A",
        "nmls",
        # "A",
        "test_acc",
        "loss",
        "test_loss",
        "W0",
        "W",
        "quad",
        "gradW",
        "harm",
        "acc",
        "rel_flatness",
    ]
    for run_id in run_ids:
        if run_id == 427:
            continue
        try:
            with open(f"run/{dataset}/{run_id}/config.json") as f:
                config = json.load(f)
            data = {}
            print(run_id)
            for list_name in data_list:
                if list_name is None:
                    continue
                data[list_name] = np.load(
                    f"res/{dataset}/{run_id}/"
                    + list_name
                    + "_list"
                    + str(run_id)
                    + ".npy"
                )
                data[list_name] = data[list_name][-1]
        except Exception as e:
            print(e)
            continue
        print(run_id)
        # data["test_loss"] *= 10
        # assert np.mean(-data["test_acc"] + data["acc"]) > 5
        # breakpoint()
        for list_name in data_list[:5]:
            if stat.get(list_name) is None:
                stat[list_name] = []
            # result = stats.pearsonr(data[list_name], data["test_loss"])
            stat[list_name].append(np.mean(data[list_name]))
            if list_name == "dim" and np.isnan(data[list_name]):
                breakpoint()
        # stat["INMLS"] = [] if stat.get("INMLS") is None else stat["INMLS"]
        # stat["INMLS"].append(np.mean(data['sharpness']*data['W']))
        stat["gen gap"] = [] if stat.get("gen gap") is None else stat["gen gap"]
        stat["gen gap"].append(np.mean(-data["test_acc"] + data["acc"]))
        # stat["gen gap"].append(np.mean(data["test_loss"] - data["loss"]))
        # stat["gen gap"].append(np.mean(data["test_acc"]))

        # stat["C"] = [] if stat.get("C") is None else stat["C"]
        stat["D"] = [] if stat.get("D") is None else stat["D"]
        # stat["C"].append(np.mean(data["gradW"]*data['quad']))
        stat["D"].append(np.mean(data["sharpness"] * data["W0"] * data["quad"]))
        # stat["D"].append(np.mean(data["sharpness"]*data["W0"]))

        stat["bound"] = [] if stat.get("bound") is None else stat["bound"]
        stat["relative flatness"] = (
            [] if stat.get("relative flatness") is None else stat["relative flatness"]
        )
        stat["bound"].append(np.mean(data["sharpness"]*np.sqrt(data["harm"])))
        # stat["bound"].append(np.mean(data["sharpness"] * data["W"]))
        stat["relative flatness"].append(np.mean(data["rel_flatness"]))

    for key in stat.keys():
        if np.any(np.isinf(stat[key])) or np.any(np.isnan(stat[key])):
            print(np.where(np.isinf(stat[key])))
            print(np.where(np.isnan(stat[key])))
            breakpoint()
    df = pd.DataFrame(stat)
    df.columns = (
        [
            "Local dim",
            "Sharpness",
            "Log volume",
            "MLS",
            "NMLS",
            "gen gap",
            "D",
            "bound",
            "relative flatness",
        ]
        # else ["Local dim", "Sharpness", "Log volume", "G", "MLS", "gen gap", "C", "D"]
    )

    # breakpoint()
    # df = pd.melt(
    #     df,
    #     id_vars=["test_loss"],
    #     value_vars=["dim", "sharpness", "logvol", "g", "A"],
    #     var_name="Metrics",
    #     value_name="Value",
    # )
    sns.set_theme(font_scale=2)
    # g = sns.scatterplot(
    #     data=df,
    #     x="test_loss",
    #     y="Value",
    #     col="Metrics",
    #     palette="tab10",
    # )
    g = sns.pairplot(df, height=3, plot_kws={"s": 80})
    g.map_lower(corrfunc)
    savefig(
        f"./image/corr/{dataset}",
        f"{dataset}_{config['network']}corr_{start_id}_to_{end_id}",
        format="pdf",
        include_timestamp=True,
    )
    # breakpoint()
