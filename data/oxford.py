from pathlib import Path
import copy
from fastai.vision import *
from fastai.metrics import *

def oxford_iiit_pet(batch_size,imgsize=224,**kwargs):
    path = untar_data(URLs.PETS)
    pat = r'/([^/]+)_\d+.jpg$'
    fnames = get_image_files(path / 'images')
    dataset = ImageDataBunch.from_name_re(path, fnames, pat, ds_tfms=get_transforms(), size=224).normalize(imagenet_stats)
    return dataset

def oxford_102_flowers(batch_size,imgsize=224,**kwargs):
    path = untar_data(URLs.FLOWERS)
    dataset = (ImageDataBunch.from_folder(path)
               .split_by_folder()
               .label_from_folder()
               .add_test_folder('test')
               .transform(tfms=get_transforms(),size=imgsize)
               .databunch(bs=batch_size,**kwargs)
               .normalize(imagenet_stats))
    return dataset
