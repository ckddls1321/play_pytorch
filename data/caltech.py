from pathlib import Path
from fastai.vision import *
from fastai.metrics import *

def caltech101(data_root=Path.home() / 'projects/DL/DB',batch_size=512,imgsize=32, **kwargs):
    # path = untar_data(URLs.CALTECH_101)
    path = url2path(URLs.CALTECH_101)
    # print(path)
    path = path.parent.joinpath('101_ObjectCategories')
    dataset = (ImageDataBunch.from_folder(path, valid_pct=0.2, ds_tfms=get_transforms(), size=imgsize).normalize(imagenet_stats))
    return dataset

def CUB200_2011(batch_size=128,imgsize=224, **kwargs):
    path = url2path(URLs.CUB_200_2011)
    dataset = (ImageDataBunch.from_folder(path / 'images', valid_pct=0.2,ds_tfms=get_transforms(), size=imgsize,bs=batch_size, **kwargs).normalize(imagenet_stats))
    return dataset

