from pathlib import Path
import copy
from fastai.vision import *
from fastai.datasets import *
from fastai.metrics import *

__all__ =['coco_tiny','coco_2017', 'coco_2014']

coco_stat =([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])

def coco_tiny(batch_size=8, imgsize=300, **kwargs):
    # path = untar_data(URLs.COCO_TINY)
    path = url2path(URLs.COCO_TINY)
    images, lbl_bbox = get_annotations(path / 'train.json')
    img2bbox = dict(zip(images, lbl_bbox))
    get_y_func = lambda o: img2bbox[o.name]
    data = (ObjectItemList.from_folder(path)
            .split_by_rand_pct()
            .label_from_func(get_y_func)
            .transform(get_transforms(), tfm_y=True, size=imgsize)
            .databunch(bs=batch_size, collate_fn=bb_pad_collate).normalize(coco_stat))

    return data

def coco_2014(batch_size=8, imgsize=300, **kwargs):
    path = Path('/datasets/coco/')
    train_images, train_lbl_bbox = get_annotations(path / 'annotations/instances_train2014.json')
    val_images, val_lbl_bbox = get_annotations(path / 'annotations/instances_val2014.json')
    images, lbl_bbox = train_images + val_images, train_lbl_bbox + val_lbl_bbox
    img2bbox = dict(zip(images, lbl_bbox))
    get_y_func = lambda o: img2bbox[o.name]

    data = (ObjectItemList.from_folder(path)
            .split_by_files(val_images)
            .label_from_func(get_y_func)
            .transform(get_transforms(), tfm_y=True, size=imgsize)
            .databunch(bs=batch_size, collate_fn=bb_pad_collate).normalize(coco_stat))

    return data


def coco_2017(batch_size=8, imgsize=300, **kwargs):
    path = Path('/datasets/coco/')
    train_images, train_lbl_bbox = get_annotations(path / 'annotations/instances_train2017.json')
    val_images, val_lbl_bbox = get_annotations(path / 'annotations/instances_val2017.json')
    images, lbl_bbox = train_images + val_images, train_lbl_bbox + val_lbl_bbox
    img2bbox = dict(zip(images, lbl_bbox))
    get_y_func = lambda o: img2bbox[o.name]

    # import shutil
    # target_path = Path('/datasets/coco/')
    # for img in train_images:
    #     shutil.copy(str(path/'train2017'/img),str(target_path / 'train'/ img))
    # for img in val_images:
    #     shutil.copy(str(path/'val2017'/img),str(target_path / 'valid'/ img))
    # print("All copied")
    # exit()

    data = (ObjectItemList.from_folder(path)
            .split_by_files(val_images)
            .label_from_func(get_y_func)
            .transform(get_transforms(), tfm_y=True, size=imgsize)
            .databunch(bs=batch_size, collate_fn=bb_pad_collate).normalize(coco_stat))

    return data
