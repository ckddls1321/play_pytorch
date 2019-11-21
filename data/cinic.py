from pathlib import Path
import copy
from fastai.vision import *
from fastai.metrics import *

def cinic10(data_root=Path('/datasets'),batch_size=512,imgsize=32, **kwargs):
    path = data_root / 'cinic10/'
    dataset = (ImageDataBunch.from_folder(path)
               .split_by_folder()
               .label_from_folder()
               .add_test_folder('test')
               .transform(tfms=get_transforms(), size=imgsize)
               .databunch(bs=batch_size, **kwargs)
               .normalize(cifar_stats))

    return dataset


