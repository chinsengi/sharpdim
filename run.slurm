#!/bin/bash

#SBATCH --job-name=sharpdim
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sc256@uw.edu
# SBATCH --account=amath
#SBATCH --account=deepthought
# SBATCH --partition=gpu-rtx6k 
#SBATCH --partition=gpu-2080ti
# SBATCH --partition=ckpt
#SBATCH --nodes=4
#SBATCH --mem=20G
#SBATCH --gres=gpu:1
#SBATCH --time=4-00:00:00

#SBATCH --chdir=.
#SBATCH --output=./slurm_out/slurm-%j.out


declare -a lr=(".05" ".1" ".2")
declare -a bs=("8" "20" "32")
# declare -a lr=(".1") 
# declare -a bs=("32")

RUNID=5
# NETWORK="fnn"
# DATASET="fashionmnist"
NETWORK="vgg"
DATASET="cifar10"
REPEAT=1
for i in ${lr[@]}; do
    for ((k = 1; k <= $REPEAT; k++)); do
        srun --ntasks=1 --gres=gpu:1 python train.py --network $NETWORK --dataset $DATASET --loss mse --run_id $RUNID --n_iters 600000 --lr $i --batch_size 20  --random &
        ((RUNID++))
    done
done

for j in ${bs[@]}; do
    if [ "$j" = "20" ]; then
        continue
    fi
    for ((k = 1; k <= $REPEAT; k++)); do
        srun --ntasks=1 --gres=gpu:1 python train.py --network $NETWORK --dataset $DATASET --loss mse --run_id $RUNID --n_iters 600000 --lr .1 --batch_size $j --random &
        ((RUNID++))
    done
done
wait
# srun --ntasks=1 --gres=gpu:1 python plot.py --run_id $RUNID
