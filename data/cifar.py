from pathlib import Path
import copy
from fastai.vision import *
from fastai.metrics import *
import torch
from torch.utils import data
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data.distributed import DistributedSampler
from utils import CIFAR10Policy

def cifar10(data_root, batch_size=512,imgsize=32, **kwargs):
    tfms = ([*rand_pad(4, imgsize), flip_lr(p=0.5)],[])
    workers = min(16, num_cpus())
    dataset = ImageDataBunch.from_folder(data_root+'cifar-10/',valid='test',ds_tfms=tfms,bs=batch_size,num_worksers=workers).normalize(cifar_stats)
    return dataset

def cifar100(data_root, batch_size=512,imgsize=32, **kwargs):
    tfms = ([*rand_pad(4, imgsize), flip_lr(p=0.5)],[])
    workers = min(16, num_cpus())
    dataset = ImageDataBunch.from_folder(data_root+'cifar-100/',valid='test',ds_tfms=tfms,bs=batch_size,num_worksers=workers).normalize(cifar_stats)
    return dataset

def cifar10_pytorch(data_root, batch_size, imgsize=32, is_distributed=False):
    workers = min(16, num_cpus())
