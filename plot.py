import matplotlib.pyplot as plt
import numpy as np
from src.utils import create_dir
import os

def plot_dim(dim_list, title, save_path, file_name):
    create_dir(save_path)
    plt.figure()
    plt.plot(dim_list)
    plt.title(title)
    plt.xlabel('iteration')
    # plt.ylabel('dimension')
    plt.savefig(os.path.join(save_path,file_name))
    # plt.close()

if __name__ == '__main__':
    run_id = "1"
    # dim_list = np.load(f'res/{run_id}/' + "dim_list" + run_id + '.npy')
    # plot_dim(dim_list, 'dimensionality', "./res/image/", "dim_list" + run_id + '.png')
    # sharpness_list = np.load(f'res/{run_id}/' + "sharpness_list" + run_id + '.npy')
    # plot_dim(sharpness_list, 'sharpness', "./res/image/", "sharpness_list" + run_id + '.png')
    logvol_list = np.load(f'res/{run_id}/' + "logvol_list" + run_id + '.npy')
    plot_dim(logvol_list, 'log volume', "./res/image/", "logvol_list" + run_id + '.png')