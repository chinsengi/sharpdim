#!/bin/bash

#SBATCH --job-name=sharpdim
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sc256@uw.edu
#SBATCH --account=deepthought
#SBATCH --partition=gpu-2080ti
#SBATCH --nodes=1
#SBATCH --mem=20G
#SBATCH --gres=gpu:1
#SBATCH --time=7-00:00:00

#SBATCH --chdir=.
#SBATCH --output=./slurm_out/slurm-%j.out


declare -a lr=(".05" ".1" ".2")
declare -a bs=("8" "20" "32")
# declare -a lr=(".1") 
# declare -a bs=("32")

RUNID=50
# NETWORK="fnn"
# DATASET="fashionmnist"
NETWORK="vgg"
DATASET="cifar10"
srun --ntasks=1 --gres=gpu:1 python train.py --network $NETWORK --dataset $DATASET --loss mse --run_id 50 --n_epochs 500 --lr .075 --batch_size 20  --random &
# srun --ntasks=1 --gres=gpu:1 python train.py --network $NETWORK --dataset $DATASET --loss mse --run_id 9 --n_epochs 500 --lr .1 --batch_size 32  --random &
    

wait
