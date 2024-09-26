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

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

def main():
    lists = ["mls_list", "sharpness_list", "norm_mls_list"]
    model = "vit_small"
    run_id = "2"

    data = {}
    for list_name in lists:
        print(f"Loading {list_name}...")
        data[list_name] = np.load(f"../run/{model}/{run_id}/{list_name}{run_id}.npy")
    mls_list = data["norm_mls_list"]
    sharpness_list = data["sharpness_list"]
    # mls_list = [mls for i, mls in enumerate(mls_list) if abs(sharpness_list[i]) < 2e7]
    # sharpness_list = [sharpness for sharpness in sharpness_list if abs(sharpness) < 2e7]
    plt.scatter(mls_list, sharpness_list)
    plt.xlabel("norm MLS")
    plt.ylabel("Sharpness")
    plt.title("MLS vs Sharpness")
    correlation, _ = pearsonr(mls_list, sharpness_list)
    print(f"Pearson correlation between MLS and Sharpness: {correlation}")
    plt.annotate(
        f"Pearson correlation: {correlation:.2f}",
        xy=(0.05, 0.95),
        xycoords="axes fraction",
        fontsize=12,
        verticalalignment="top",
    )
    savefig(f"../run/{model}/{run_id}")


if __name__ == "__main__":
    main()
