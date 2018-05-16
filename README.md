# GANomaly

This repository contains PyTorch implementation of **"PaperName"**.

## Task List
- [ ] Update options - (Remove the ones not used.)

##  Table of Contents
- [GANomaly](#ganomaly)
    - [Task List](#task-list)
    - [Table of Contents](#table-of-contents)
    - [Prerequisites](#prerequisites)
    - [Experiment](#experiment)
    - [Training](#training)


## Prerequisites
1. OS: Linux or MacOS
2. PyTorch v0.3 - For now v0.4 is not supported
3. GPU - Highly recommended


## Experiment

To replicate the results in the paper, run the following commands:

For MNIST experiments:
``` shell
sh run_mnist.sh
```

For CIFAR experiments:
``` shell
sh run_cifar.sh
```

## Training
To get to know the arguments to train the model, run the following:
```
python train.py -h

usage: train.py

    -h, --help          Show this help message and exit
    --dataset           Mnist | cifar10 | folder (default: mnist)
    --dataroot          Path to dataset (default: )
    --batchsize         Input batch size (default: 64)
    --workers           Number of data loading workers (default: 8)
    --droplast          Drop last batch size. (default: True)
    --isize             Input image size. (default: 32)
    --nc                Input image channels (default: 3)
    --nz                Size of the latent z vector (default: 100)
    --ngf               Number of features - generator network (default: 64)
    --ndf               Number of features - discriminator network (default: 64)
    --extralayers       Number of extra layers on gen and disc (default: 0)
    --gpu_ids           gpu ids: e.g. 0 0,1,2, 0,2. use -1 for CPU (default:0)
    --ngpu              Number of GPUs to use (default: 1)
    --name              Name of the experiment (default: experiment_name)
    --model             Chooses which model to use. ganomaly (default:ganomaly)
    --display_server    Visdom server of the web display (default:http://localhost)
    --display_port      Visdom port of the web display (default: 8097)
    --display_id        Window id of the web display (default: 0)
    --display           Use visdom. (default: False)
    --outf              Folder to output images and model checkpoints (default: ./output)
    --manualseed        Manual seed (default: None)
    --anomaly_class     Anomaly class idx for mnist and cifar datasets (default: 0)
    --display_freq      Frequency of showing training results on screen (default: 100)
    --print_freq        Frequency of showing training results on console (default: 100)
    --save_latest_freq  Frequency of saving the latest results (default: 5000)
    --save_epoch_freq   Frequency of saving checkpoints at the end of epochs (default: 1)
    --save_image_freq   Frequency of saving real and fake images (default: 100)
    --save_test_images  Save test images for demo. (default: False)
    --load_weights      Load the pretrained weights (default: False)
    --resume            Path to checkpoints (to continue training) (default: )
    --phase             Train, val, test, etc (default: train)
    --iter              Start from iteration i (default: 0)
    --niter             Number of epochs to train for (default: 15)
    --beta1             Momentum term of adam (default: 0.5)
    --lr LR             Initial learning rate for adam (default: 0.0002)
    --lr_policy         Learning rate policy: lambda|step|plateau (default: lambda)
    --lr_decay_iters    Multiply by a gamma every lr_decay_iters iterations  (default: 50)
    --gen_type          Type of the generator network. bowtie | regular (default: bowtie)
    --dct               Add DCT to the GAN loss. (default: False)
    --alpha             Alpha to weight l1 loss. default=500 (default: 50)
```



```
python train.py --dataset <name-of-the-data> --isize <image-size>
```