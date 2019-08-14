"""
TRAIN GANOMALY

. Example: Run the following command from the terminal.
    run train.py                             \
        --model ganomaly                        \
        --dataset UCSD_Anomaly_Dataset/UCSDped1 \
        --batchsize 32                          \
        --isize 256                         \
        --nz 512                                \
        --ngf 64                               \
        --ndf 64
"""


##
# LIBRARIES
from __future__ import print_function

from options import Options
from lib.data import load_data
from lib.model import Ganomaly

##
def train():
    """ Training
    """

    ##
    # ARGUMENTS
    opt = Options().parse()
    ##
    # LOAD DATA
    dataloader = load_data(opt)
    ##
    # LOAD MODEL
    model = Ganomaly(opt, dataloader)
    ##
    # TRAIN MODEL
    model.train()

if __name__ == '__main__':
    train()
