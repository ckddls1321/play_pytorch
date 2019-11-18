from pathlib import Path
from fastai.vision import *
from fastai.metrics import *

def stanford_cars(batch_size=256,imgsize=224, **kwargs):
    # path = untar_data(URLs.CARS)
    path = url2path(URLs.MNIST)
    dataset = ImageDataBunch.from_folder(path,valid='test',ds_tfms=get_transforms(),bs=batch_size).normalize(imagenet_stats)
    return dataset

