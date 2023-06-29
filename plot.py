import matplotlib.pyplot as plt
import numpy as np

def plot_dim(dim_list, title, save_path):
    plt.figure()
    plt.plot(dim_list)
    plt.title(title)
    plt.xlabel('iteration')
    # plt.ylabel('dimension')
    plt.savefig(save_path)
    # plt.close()

if __name__ == '__main__':
    run_id = "0"
    dim_list = np.load('res/' + "dim_list" + run_id + '.npy')
    plot_dim(dim_list, 'dimensionality', "./res/image/dim_list" + run_id + '.png')
    sharpness_list = np.load('res/' + "sharpness_list" + run_id + '.npy')
    plot_dim(sharpness_list, 'sharpness', "./res/image/sharpness_list" + run_id + '.png')
    logvol_list = np.load('res/' + "logvol_list" + run_id + '.npy')
    plot_dim(logvol_list, 'log volume', "./res/image/logvol_list" + run_id + '.png')