import matplotlib.pyplot as plt
import numpy as np

def plot_dim(dim_list, title, save_path):
    plt.figure()
    plt.plot(dim_list)
    plt.title(title)
    plt.xlabel('iteration')
    plt.ylabel('dimension')
    plt.savefig(save_path)
    plt.close()

if __name__ == '__main__':
    dim_list = np.load('res/dim_list.npy')
    plot_dim(dim_list, 'dimensionality', 'dim_list.png')