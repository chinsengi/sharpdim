# Sharpness and Dimensionality in Neural Networks

## Hyak commands
`salloc -A amath -p gpu-rtx6k -N 1 -c 1 --mem=40G  --time=24:00:00 --gpus=1` 
`squeue -u <net_id>`
`python train.py --network fnn --dataset fashionmnist --loss mse --run_id 0 --n_iters 150000 --lr .1 --batch_size 8 --random --cal_freq 50`