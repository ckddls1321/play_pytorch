from pathlib import Path
import copy
from fastai.vision import *
from fastai.metrics import *

def food101(batch_size=256, imgsize=224, **kwargs):
    # path = untar_data(URLs.FOOD)
    path = url2path(URLs.FOOD)
    dataset = (ImageDataBunch.from_folder(path,valid_pct=0.2,test='test',ds_tfms=get_transforms(),size=imgsize,bs=batch_size,**kwargs).normalize(imagenet_stats))
    return dataset

def fruits360(data_root = Path.home() / 'projects/DL/DB', batch_size=256, imgsize=224, **kwargs):
    path = data_root / 'fruits-360/'
    dataset = (ImageDataBunch.from_folder(path,valid_pct=0.2,test='test',ds_tfms=get_transforms(),size=imgsize,bs=batch_size,**kwargs).normalize(imagenet_stats))
    return dataset
