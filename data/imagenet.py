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

def imagenet(batch_size=32,imgsize=224, **kwargs):
    path = Path('/datasets/imagenet')
    dataset = ImageDataBunch.from_folder(path,valid='val',ds_tfms=get_transforms(),bs=batch_size,size=imgsize).normalize(imagenet_stats)
    return dataset

def imagenet64(batch_size=128,imgsize=64, **kwargs):
    path = Path('/datasets/imagenet64')
    dataset = ImageDataBunch.from_folder(path,valid='val',ds_tfms=get_transforms(),bs=batch_size,size=imgsize).normalize(imagenet_stats)
    return dataset
