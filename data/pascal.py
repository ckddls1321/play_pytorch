from pathlib import Path
import copy
from fastai.vision import *
from fastai.metrics import *
import torch
from torch.utils import data
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data.distributed import DistributedSampler

__all__ = ['pascal_2007', 'pascal_2012']

pascal_stat = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

def pascal_2007(batch_size=16, imgsize=256, **kwargs):
    path = untar_data(URLs.PASCAL_2007)
    train_images, train_lbl_bbox = get_annotations(path / 'pascal_train2007.json')
    val_images, val_lbl_bbox = get_annotations(path / 'pascal_val2007.json')
    images, lbl_bbox = train_images + val_images, train_lbl_bbox + val_lbl_bbox
    img2bbox = dict(zip(images, lbl_bbox))
    get_y_func = lambda o: img2bbox[o.name]

    data = (ObjectItemList.from_folder(path / '2007' / 'train')
            .split_by_files(val_images)
            .label_from_func(get_y_func)
            .transform(get_transforms(), tfm_y=True, size=imgsize)
            .databunch(bs=batch_size, collate_fn=bb_pad_collate).normalize(pascal_stat))

    return data

def pascal_2012(batch_size=16, imgsize=256, **kwargs):
    path = untar_data(URLs.PASCAL_2012)
    train_images, train_lbl_bbox = get_annotations(path / 'pascal_train2012.json')
    val_images, val_lbl_bbox = get_annotations(path / 'pascal_val2012.json')
    images, lbl_bbox = train_images + val_images, train_lbl_bbox + val_lbl_bbox
    img2bbox = dict(zip(images, lbl_bbox))
    get_y_func = lambda o: img2bbox[o.name]

    data = (ObjectItemList.from_folder(path)
            .split_by_files(val_images)
            .label_from_func(get_y_func)
            .transform(get_transforms(), tfm_y=True, size=imgsize)
            .databunch(bs=batch_size, collate_fn=bb_pad_collate).normalize(pascal_stat))

    return data
