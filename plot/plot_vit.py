import logging
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

# Add the parent directory of src to the sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.utils import create_dir, savefig
import argparse
import pandas as pd
import seaborn as sns
import json
from scipy.signal import savgol_filter
import scienceplots as sp
import math
from scipy.stats import pearsonr
import warnings
from pprint import pprint

warnings.filterwarnings("ignore")

def main():
    logger = logging.getLogger()
    logger.addHandler(logging.StreamHandler())
    logger.setLevel(logging.INFO)
    lists = ["mls_list", "sharpness_list", "norm_mls_list", "model_name_list"]
    model = "vit"
    run_id = "6"

    data = {}
    for list_name in lists:
        print(f"Loading {list_name}...")
        data[list_name] = np.load(f"../run/{model}/{run_id}/{list_name}{run_id}.npy")
    norm_mls_list = np.array(data["norm_mls_list"])
    mls_list = np.array(data["mls_list"])
    sharpness_list = np.array(data["sharpness_list"])
    model_name_list = data["model_name_list"]
    pprint(model_name_list)
    model_list = [name.split('_')[0] for name in model_name_list]
    
    #delete test model
    test_index = np.where(np.array(model_list) == "test")
    model_list = np.delete(model_list, test_index)
    norm_mls_list = np.delete(norm_mls_list, test_index)
    sharpness_list = np.delete(sharpness_list, test_index)
    mls_list = np.delete(mls_list, test_index)
    model_name_list = np.delete(model_name_list, test_index)
    
    unique_name = np.unique(model_list)
    string_to_number = {string: i for i, string in enumerate(unique_name)}
    model_id = [string_to_number[name] for name in model_list]
    pprint([name for i, name in enumerate(model_name_list) if norm_mls_list[i]<20000 and sharpness_list[i]>1000])
    pprint([name for i, name in enumerate(model_name_list) if norm_mls_list[i]>60000 and sharpness_list[i]<1000])
    logging.warning(f"model num: {len(mls_list)}")
    coefficients = np.polyfit(norm_mls_list, sharpness_list, 2)
    # Generate points for plotting the quadratic curve
    x_fit = np.linspace(min(norm_mls_list), max(norm_mls_list), 100)
    y_fit = np.polyval(coefficients, x_fit)

    # Plot the data points and the fitted quadratic curve
    plt.scatter(norm_mls_list, sharpness_list, color='red', label='Data points')
    plt.plot(x_fit, y_fit, label=f'Quadratic fit: {coefficients[0]:.2f}x^2 + {coefficients[1]:.2f}x + {coefficients[2]:.2f}')
    plt.xlabel('norm MLS')
    plt.ylabel('Sharpness')
    plt.title('Quadratic Function Fit')
    plt.legend()
    plt.grid(True)
    # savefig(f"../run/{model}/{run_id}", "quadratic_fit")
    plt.figure()
    plt.scatter(mls_list, sharpness_list)
    plt.xlabel("MLS")
    plt.ylabel("Sharpness")
    plt.title("MLS vs Sharpness")
    correlation, _ = pearsonr(mls_list, sharpness_list)
    print(f"Pearson correlation between MLS and Sharpness: {correlation}")
    # plt.annotate(
    #     f"Pearson correlation: {correlation:.2f}",
    #     xy=(0.05, 0.95),
    #     xycoords="axes fraction",
    #     fontsize=12,
    #     verticalalignment="top",
    # )
    # savefig(f"../run/{model}/{run_id}", "mls_vs_sharpness")
    plt.figure()
    norm_correlation, _ = pearsonr(norm_mls_list, sharpness_list)
    logging.warning(
        f"Pearson correlation between Normalized MLS and Sharpness: {norm_correlation}"
    )
    plt.scatter(norm_mls_list, sharpness_list, marker="x", c=model_id, cmap='tab10')
    plt.xlabel("Normalized MLS")
    plt.ylabel("Adaptive Sharpness")
    plt.grid(True)
    plt.yscale("log")
    # plt.title("MLS vs Sharpness")
    # plt.annotate(
    #     f"Pearson correlation: {norm_correlation:.2f}",
    #     xy=(0.05, 0.95),
    #     xycoords="axes fraction",
    #     fontsize=12,
    #     verticalalignment="top",
    # )

    savefig(f"../run/{model}/{run_id}", "norm_mls_vs_sharpness")
    


if __name__ == "__main__":
    main()
