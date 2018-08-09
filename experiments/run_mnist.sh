#!/bin/bash

# Run MNIST experiment for each individual dataset.
# For each anomalous digit
for m in {0..2}
do
    echo "Manual Seed: $m"
    for i in {0..9}
    do
        echo "Running mnist_$i"
        python train.py --dataset mnist --isize 32 --nc 1 --niter 15 --anomaly_class $i --manualseed $m --display
    done
done
exit 0
