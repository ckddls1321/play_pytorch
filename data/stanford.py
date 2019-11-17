from pathlib import Path
from fastai.vision import *
from fastai.metrics import *

def stanford_cars(batch_size=256,imgsize=224, **kwargs):
    tfms = get_transforms()
    path = untar_data(URLs.CARS)
    dataset = ImageDataBunch.from_folder(path,valid='test',ds_tfms=tfms,bs=batch_size).normalize(imagenet_stats)
    return dataset

