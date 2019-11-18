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

def cifar10(batch_size=512,imgsize=32, **kwargs):
    tfms = ([*rand_pad(4, imgsize), flip_lr(p=0.5)],[])
    # path = untar_data(URLs.CIFAR)
    path = url2path(URLs.CIFAR)
    dataset = ImageDataBunch.from_folder(path,valid='test',ds_tfms=tfms,bs=batch_size).normalize(cifar_stats)
    return dataset

def cifar100(batch_size=512,imgsize=32, **kwargs):
    tfms = ([*rand_pad(4, imgsize), flip_lr(p=0.5)],[])
    # path = untar_data(URLs.CIFAR_100)
    path = url2path(URLs.CIFAR_100)
    dataset = ImageDataBunch.from_folder(path,valid='test',ds_tfms=tfms,bs=batch_size).normalize(cifar_stats)
    return dataset

def cifar10_pytorch(data_root, batch_size, imgsize=32, is_distributed=False):
    workers = min(16, num_cpus())
