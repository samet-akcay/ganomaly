#!/bin/bash

# Run CIFAR10 experiment on ganomaly

declare -a arr=("plane" "car" "bird" "cat" "deer" "dog" "frog" "horse" "ship" "truck" )
for m in $(seq 0 2)
do
    echo "Manual Seed: $m"
    for i in "${arr[@]}";
    do
        echo "Running CIFAR. Anomaly Class: $i "
        python train.py --dataset cifar10 --isize 32 --niter 15 --abnormal_class $i --manualseed $m
    done
done
exit 0
