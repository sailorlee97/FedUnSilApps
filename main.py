# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


import torch
import numpy as np
from unlearning import unlearn
import sys
from utils import *
import argparse

# torch.cuda.device_count()
# os.environ['CUDA_VISIBLE_DEVICES'] = '%d' % m_gpu
# torch.cuda.set_device(m_gpu)
# torch.cuda.is_available()
# torch.cuda.current_device()
parser = argparse.ArgumentParser(description='Machine Unlearning')
parser.add_argument('--batch_size', default = 256, type = int)
parser.add_argument('--epoch', default = 5, type = int)
parser.add_argument('--lr', default = 0.001, type = int)
parser.add_argument('--max_size', default = 8000, type = int)
parser.add_argument('--total_cls', default = 10, type = int)
parser.add_argument('--incremental_list', default = [10,0], type = list)
args = parser.parse_args()


if __name__ == "__main__":
    # showGod()
    trainer = unlearn(args.total_cls,args.incremental_list)
    trainer.train(args.batch_size, args.epoch, args.lr, args.max_size)
