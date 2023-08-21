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
    if os.path.exists(f"res/df_{dataset}.pkl") and not recompute:
        return pd.read_pickle(f"res/df_{dataset}.pkl")
    dfs = []
    for run_id in run_ids:
        if run_id == 5:
            continue
        # load the json file
        with open(f"run/{run_id}/config.json") as f:
            config = json.load(f)
        data_list = [
            "dim_list",
            "sharpness_list",
            "logvol_list",
            "acc_list",
            "g_list",
            "eig_list",
            "loss_list",
        ]
        data = {}
        for list_name in data_list:
            data[list_name] = np.load(
                f"res/{run_id}/" + list_name + str(run_id) + ".npy"
            )
        train_size = 60000 if config["dataset"] == "fashionmnist" else 10000
        batch_size = config["batch_size"]
        n_iter_per_epoch = train_size // batch_size
        list_len = len(data["dim_list"])
        print(f"batch_size: {config['batch_size']}, list_len: {list_len}")
        sample_freq = config["n_iters"] / list_len
        # if run_id ==50:
        #     breakpoint()
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
                "eig": data["eig_list"].tolist(),
            }
        )
        dfs.append(df)
    dfs = pd.concat(dfs)
    vars2idxs = ['epoch', 'network', 'dataset', 'iteration', 'lr', 'batch_size']
    vars2stack = ['train_accuracy', 'train_loss', 'log_volume', 'dimension', 'sharpness', 'log_volume', 'G']
    df = dfs.set_index(vars2idxs)[vars2stack]
    df = df.stack().reset_index().rename(columns={0:'value', 'level_'+str(len(vars2idxs)):'variable'})
    dfs.to_pickle(f"res/df_{dataset}.pkl")
    return df


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
        run_ids = [i for i in range(10)] + [10 + 5*i for i in range(2)] + [50] if dataset == "cifar10" else range(21, 26)
        df = construct_df(run_ids, dataset, recompute=True)
        breakpoint()
        df1 = df[df["batch_size"] == 20]
        # df1['iteration_epoch'] = df1['iteration'] -  df1['epoch']*100
        df1['pcg_training'] = 100*round(df1['iteration'] / df1['iteration'].max() * 30,1)/30  
        g = sns.relplot(data=df1, x='pcg_training', y='value', hue='lr', col='variable', kind='line', facet_kws={'sharey':False})
        sns.despine()
        for item, ax in g.axes_dict.items():
            ax.grid(False, axis='x')
            ax.set_title(item)  
        savefig('./res/image', "vgg10_cifar_batch20", format='png')

        dfs = df[df['lr']==0.1]
        # dfs['iteration_epoch'] = dfs['iteration'] -  dfs['epoch']*100
        dfs['pcg_training'] = 100*round(dfs['iteration'] / dfs['iteration'].max() * 30,1)/30
        g = sns.relplot(data=dfs, x='pcg_training', y='value', hue='batch_size', col='variable', kind='line', facet_kws={'sharey':False})
        sns.despine()
        for item, ax in g.axes_dict.items():
            ax.grid(False, axis='x')
            ax.set_title(item)
        savefig('./res/image', "vgg10_cifar_lr01", format='png')


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
