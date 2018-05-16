#!/bin/bash

# Run CIFAR10 experiment on ganomaly

declare -a arr=("plane" "car" "bird" "cat" "deer" "dog" "frog" "horse" "ship" "truck" )
for i in "${arr[@]}";
do
    echo "Running CIFAR. Anomaly Class: $i "
    python train.py --dataset cifar10 --isize 32 --niter 15 --anomaly_class $i
done
exit 0
