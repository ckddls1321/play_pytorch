from pathlib import Path
from fastai.vision import *
from fastai.metrics import *

def caltech_101(data_root=Path.home() / 'projects/DL/DB',batch_size=512,imgsize=32, **kwargs):
    tfms = ([*rand_pad(4, imgsize), flip_lr(p=0.5)],[])
    path = untar_data(URLs.CALTECH_101)
    dataset = ImageDataBunch.from_folder(path,valid='test',ds_tfms=tfms,bs=batch_size).normalize(imagenet_stats)
    return dataset

def CUB_200_2011(batch_size=512,imgsize=32, **kwargs):
    tfms = ([*rand_pad(4, imgsize), flip_lr(p=0.5)],[])
    path = untar_data(URLs.CUB_200_2011)
    dataset = ImageDataBunch.from_folder(path,valid='test',ds_tfms=tfms,bs=batch_size).normalize(imagenet_stats)
    return dataset

