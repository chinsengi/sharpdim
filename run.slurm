#!/bin/bash

#SBATCH --job-name=gen_RNN
#SBATCH --mail-type=END
#SBATCH --mail-user=sc256@uw.edu
#SBATCH --account=amath
#SBATCH --partition=gpu-rtx6k 
# SBATCH --partition=ckpt
#SBATCH --nodes=1
#SBATCH --mem=20G
#SBATCH --gres=gpu:1

#SBATCH --chdir=.
#SBATCH --output=./slurm-%j.out

srun --ntasks=1 --gres=gpu:1 python train.py --dataset fashionmnist --n_iters 100000 
