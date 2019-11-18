from pathlib import Path
import copy
from fastai.vision import *
from fastai.metrics import *

def mnist(batch_size=512,imgsize=28, **kwargs):
    tfms = ([*rand_pad(3, imgsize, mode='zeros')],[])
    # path = untar_data(URLs.MNIST)
    path = url2path(URLs.MNIST)
    dataset = ImageDataBunch.from_folder(path,valid='test',ds_tfms=tfms,size=imgsize, bs=batch_size).normalize()
    return dataset

def fashionmnist(data_root= Path.home()/'projects/DL/DB', batch_size=512,imgsize=28, **kwargs):
    tfms = ([*rand_pad(3, imgsize, mode='zeros')],[])
    path = data_root / 'fashionmnist'
    dataset = ImageDataBunch.from_folder(path,valid='test',ds_tfms=tfms,size=imgsize, bs=batch_size).normalize()
    return dataset


