from pathlib import Path
import copy
from fastai.vision import *
from fastai.metrics import *

def fruits360(data_root = Path.home() / 'projects/DL/DB', batch_size=256, imgsize=224, **kwargs):
    tfms = get_transforms()
    path = data_root / 'fruits-360/'
    dataset = (ImageDataBunch.from_folder(path)
               .split_by_rand_pct()
               .label_from_folder()
               .add_test_folder('test')
               .transform(tfms=tfms,size=imgsize)
               .databunch(bs=batch_size,**kwargs)
               .normalize(imagenet_stats))
    return dataset
