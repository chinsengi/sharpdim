#!/bin/bash

#SBATCH --job-name=sharpdim
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sc256@uw.edu
#SBATCH --account=amath
# SBATCH --account=deepthought
#SBATCH --partition=gpu-rtx6k
# SBATCH --partition=gpu-2080ti
# SBATCH --partition=ckpt
#SBATCH --nodes=8
#SBATCH --mem=7G
#SBATCH --gres=gpu:1
#SBATCH --time=2-00:00:00

#SBATCH --chdir=.
#SBATCH --output=./slurm_out/slurm-%j.out

declare -a lr=(".05" ".1" ".2")
declare -a bs=("8" "20" "32")
# declare -a lr=(".1")
# declare -a bs=("20")

RUNID=400
NETWORK="fnn"
DATASET="fashionmnist"
NETWORK="vgg"
# NETWORK="lenet5"
# NETWORK="lenet"
DATASET="cifar10"
# N_ITERS="1000000"
N_ITERS="750000" 
# N_ITERS="500000" 
CAL_FREQ="750000"
# CAL_FREQ="500000"
REPEAT=20
for ((k = 1; k <= $REPEAT; k++)); do
    for i in ${lr[@]}; do
        srun --ntasks=1 --gres=gpu:1 python train.py --network $NETWORK --dataset $DATASET --loss mse --run_id $RUNID --n_iters $N_ITERS --lr $i --batch_size 20  --random --cal_freq $CAL_FREQ --nonlinearity tanh&
        ((RUNID++))
    done

    for j in ${bs[@]}; do
        if [ "$j" = "20" ]; then
            continue
        fi
        srun --ntasks=1 --gres=gpu:1 python train.py --network $NETWORK --dataset $DATASET --loss mse --run_id $RUNID --n_iters $N_ITERS --lr .1 --batch_size $j --random --cal_freq $CAL_FREQ --nonlinearity tanh&
        ((RUNID++))
    done
done
wait
# srun --ntasks=1 --gres=gpu:1 python plot.py --run_id $RUNID
