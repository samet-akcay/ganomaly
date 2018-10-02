# GANomaly

This repository contains PyTorch implementation of the following paper: GANomaly: Semi-Supervised Anomaly Detection via Adversarial Training [[1]](#reference)

##  Table of Contents
- [GANomaly](#ganomaly)
    - [Table of Contents](#table-of-contents)
    - [Prerequisites](#prerequisites)
    - [Experiment](#experiment)
    - [Training](#training)
        - [Training on MNIST](#training-on-mnist)
        - [Training on CIFAR10](#training-on-cifar10)
        - [Train on Custom Dataset](#train-on-custom-dataset)
    - [Citing GANomaly](#citing-ganomaly)
    - [Reference](#reference)


## Prerequisites
1. Linux or MacOS
2. Python 2 or 3
3. CPU or GPU + CUDA & CUDNN

## Installation
1. First clone the repository
   ```
   git clone https://github.com/samet-akcay/ganomaly.git
   ```
2. Install PyTorch and torchvision from [https://pytorch.org](https://pytorch.org/)
3. Install the dependencies.
   ```
   pip install -r requirements.txt
   ```
**UPDATE**: This repository now supports PyTorch v0.4. If you still would like to work with v0.3, you could use the branch named PyTorch.v0.3, which contains the previous version of the repo.

## Experiment

To replicate the results in the paper, run the following commands:

For MNIST experiments:
``` shell
sh experiments/run_mnist.sh
```

For CIFAR experiments:
``` shell
sh experiments/run_cifar.sh
```

## Training
To list the arguments, run the following command:
```
python train.py -h
```

### Training on MNIST
To train the model on MNIST dataset for a given anomaly class, run the following:

``` 
python train.py \
    --dataset mnist             \
    --niter <number-of-epochs>  \
    --anomaly_class <0,1,2,3,4,5,6,7,8,9>
```

### Training on CIFAR10
To train the model on CIFAR10 dataset for a given anomaly class, run the following:

``` 
python train.py \
    --dataset cifar10             \
    --niter <number-of-epochs>    \
    --anomaly_class               \
        <plane, car, bird, cat, deer, dog, frog, horse, ship, truck>
```

### Train on Custom Dataset
To train the model on a custom dataset, the dataset should be copied into `./data` directory, and should have the following directory & file structure:

```
Custom Dataset
├── test
│   ├── 0.normal
│   │   └── normal_tst_img_0.png
│   │   └── normal_tst_img_1.png
│   │   ...
│   │   └── normal_tst_img_n.png
│   ├── 1.abnormal
│   │   └── abnormal_tst_img_0.png
│   │   └── abnormal_tst_img_1.png
│   │   ...
│   │   └── abnormal_tst_img_m.png
├── train
│   ├── 0.normal
│   │   └── normal_tst_img_0.png
│   │   └── normal_tst_img_1.png
│   │   ...
│   │   └── normal_tst_img_t.png

```

Then model training is the same as training MNIST or CIFAR10 datasets explained above.

```
python train.py                     \
    --dataset <name-of-the-data>    \
    --isize <image-size>            \
    --niter <number-of-epochs>
```

For more training options, run `python train.py -h` as shown below:
```
usage: train.py [-h] [--dataset DATASET] [--dataroot DATAROOT]
                [--batchsize BATCHSIZE] [--workers WORKERS] [--droplast]
                [--isize ISIZE] [--nc NC] [--nz NZ] [--ngf NGF] [--ndf NDF]
                [--extralayers EXTRALAYERS] [--gpu_ids GPU_IDS] [--ngpu NGPU]
                [--name NAME] [--model MODEL]
                [--display_server DISPLAY_SERVER]
                [--display_port DISPLAY_PORT] [--display_id DISPLAY_ID]
                [--display] [--outf OUTF] [--manualseed MANUALSEED]
                [--anomaly_class ANOMALY_CLASS] [--print_freq PRINT_FREQ]
                [--save_image_freq SAVE_IMAGE_FREQ] [--save_test_images]
                [--load_weights] [--resume RESUME] [--phase PHASE]
                [--iter ITER] [--niter NITER] [--beta1 BETA1] [--lr LR]
                [--alpha ALPHA]

optional arguments:
  -h, --help            show this help message and exit
  --dataset             folder | cifar10 | mnist (default: cifar10)
  --dataroot            path to dataset (default: '')
  --batchsize           input batch size (default: 64)
  --workers             number of data loading workers (default: 8)
  --droplast            Drop last batch size. (default: True)
  --isize               input image size. (default: 32)
  --nc                  input image channels (default: 3)
  --nz                  size of the latent z vector (default: 100)
  --ngf                 Number of features of the generator network
  --ndf                 Number of features of the discriminator network.
  --extralayers         Number of extra layers on gen and disc (default: 0)
  --gpu_ids             gpu ids: e.g. 0 0,1,2, 0,2. use -1 for CPU (default: 0)
  --ngpu                number of GPUs to use (default: 1)
  --name                name of the experiment (default: experiment_name)
  --model               chooses which model to use. (default:ganomaly)
  --display_server      visdom server of the web display (default: http://localhost)
  --display_port        visdom port of the web display (default: 8097)
  --display_id          window id of the web display (default: 0)
  --display             Use visdom. (default: False)
  --outf                folder to output images and model checkpoints (default: ./output)
  --manualseed          manual seed (default: None)
  --anomaly_class       Anomaly class idx for mnist and cifar datasets (default: car)
  --print_freq          frequency of showing training results on console (default: 100)
  --save_image_freq     frequency of saving real and fake images (default:100)
  --save_test_images    Save test images for demo. (default: False)
  --load_weights        Load the pretrained weights (default: False)
  --resume              path to checkpoints (to continue training) (default: '')
  --phase               train, val, test, etc (default: train)
  --iter                Start from iteration i (default: 0)
  --niter               number of epochs to train for (default: 15)
  --beta1               momentum term of adam (default: 0.5)
  --lr                  initial learning rate for adam (default: 0.0002)
  --alpha               alpha to weight l1 loss. default=500 (default: 50)

```

## Citing GANomaly
If you use this repository or would like to refer the paper, please use the following BibTeX entry
```
@article{Akcay2018,
    author = {Akcay, S. and Atapour-Abarghouei, A. and Breckon, T.~P.},
    title = "{GANomaly: Semi-Supervised Anomaly Detection via Adversarial Training}",
    journal = {ArXiv e-prints},
    archivePrefix = "arXiv",
    eprint = {1805.06725},
    primaryClass = "cs.CV",
    keywords = {Computer Science - Computer Vision and Pattern Recognition},
    year = 2018,
    month = may,
}
```

## Reference
[[1]  S. Akcay, A. Atapour-Abarghouei, and T. P. Breckon.  GANomaly:  Semi-SupervisedAnomaly Detection via Adversarial Training. ArXiv e-prints, May 2018.](https://arxiv.org/abs/1805.06725)
