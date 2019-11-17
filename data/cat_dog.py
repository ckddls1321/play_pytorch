from pathlib import Path
import copy
from fastai.vision import *
from fastai.metrics import *

def cat_dog(batch_size,imgsize=224,**kwargs):
    tfms = get_transforms()
    path = untar_data(URLs.DOGS)
    dataset = (ImageDataBunch.from_folder(path)
               .split_by_folder()
               .label_from_folder()
               .transform(tfms=tfms,size=imgsize)
               .databunch(bs=batch_size,**kwargs)
               .normalize(cifar_stats))
    return dataset
