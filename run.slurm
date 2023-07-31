#!/bin/bash

#SBATCH --job-name=sharpdim
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sc256@uw.edu
#SBATCH --account=amath
#SBATCH --partition=gpu-rtx6k 
# SBATCH --partition=ckpt
#SBATCH --nodes=6
#SBATCH --mem=20G
#SBATCH --gres=gpu:1
#SBATCH --time=7-00:00:00

#SBATCH --chdir=.
#SBATCH --output=./slurm-%j.out


declare -a lr=(".05" ".1" ".2")
declare -a bs=("8" "16" "20")

RUNID=0
# NETWORK="fnn"
# DATASET="fashionmnist"
NETWORK="vgg"
DATASET="cifar10"
for i in ${lr[@]}; do
    for ((k = 1; k <= 5; k++)); do
        srun --ntasks=1 --gres=gpu:1 python train.py --network $NETWORK --dataset $DATASET --loss mse --run_id $RUNID --n_iters 600000 --lr $i --batch_size 20 &
        ((RUNID++))
    done
done

for j in ${bs[@]}; do
    if [ "$j" = "20" ]; then
        continue
    fi
    for ((k = 1; k <= 5; k++)); do
        srun --ntasks=1 --gres=gpu:1 python train.py --network $NETWORK --dataset $DATASET --loss mse --run_id $RUNID --n_iters 600000 --lr .1 --batch_size $j &
        ((RUNID++))
    done
done
wait
# srun --ntasks=1 --gres=gpu:1 python plot.py --run_id $RUNID
