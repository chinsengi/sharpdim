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
    datafile_name = "dim_list1"
    dim_list = np.load('res/' + datafile_name + '.npy')
    plot_dim(dim_list, 'dimensionality', datafile_name + '.png')