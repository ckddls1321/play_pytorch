from pathlib import Path
import copy
from fastai.vision import *
from fastai.metrics import *

def cat_dog(batch_size,imgsize=224,**kwargs):
    path = untar_data(URLs.DOGS)
    dataset = (ImageDataBunch.from_folder(path,ds_tfms=get_transforms(),size=imgsize,valid_pct=0.2,bs=batch_size,**kwargs).normalize(imagenet_stats))
    return dataset
