#!/bin/bash

#SBATCH --job-name=sharpdim
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sc256@uw.edu
#SBATCH --account=amath
#SBATCH --partition=gpu-rtx6k 
# SBATCH --partition=ckpt
#SBATCH --nodes=1
#SBATCH --mem=40G
#SBATCH --gres=gpu:1
#SBATCH --time=1-00:00:00

#SBATCH --chdir=.
#SBATCH --output=./slurm-%j.out


srun --ntasks=1 --gres=gpu:1 python train.py --loss mse --run_id 1 --n_iters 20000 
