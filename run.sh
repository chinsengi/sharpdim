#!/bin/bash
#learning_rate=0.1
batch_size=2

i=1
for learning_rate in 0.02 0.1 0.5
do
  for rep in 1 2 3 4 5
  do
    python train.py --dataset fashionmnist --n_iters 100000 --batch_size $batch_size --load_size $batch_size --learning_rate $learning_rate
  done
  i=$((i+1))
done




