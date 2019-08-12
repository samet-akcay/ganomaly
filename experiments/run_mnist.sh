#!/bin/bash

# Run MNIST experiment for each individual dataset.
# For each anomalous digit
for m in {0..2}
do
    echo "Manual Seed: $m"
    for i in {0..9}
    do
        echo "Running MNIST, Abnormal Digit: $i"
        python train.py --dataset mnist --isize 32 --nc 1 --niter 15 --abnormal_class $i --manualseed $m
    done
done
exit 0
