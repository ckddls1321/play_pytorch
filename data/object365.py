from pathlib import Path
import copy
from fastai.vision import *
from fastai.datasets import *
from fastai.metrics import *

__all__ =['object365','object365_tiny']

coco_stat =([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])

def object365(batch_size=8, imgsize=300, **kwargs):
    path = Path('/datasets/object365')
    train_images, train_lbl_bbox = get_annotations(path / 'objects365_train.json')
    val_images, val_lbl_bbox = get_annotations(path / 'objects365_val.json')
    images, lbl_bbox = train_images + val_images, train_lbl_bbox + val_lbl_bbox
    img2bbox = dict(zip(images, lbl_bbox))
    get_y_func = lambda o: img2bbox[o.name]

    # import shutil
    # target_path = Path('/datasets/object365')
    # for img in train_images:
    #     shutil.copy(str(path/'train'/img),str(target_path / 'train2'/ img))
    # for img in val_images:
    #     shutil.copy(str(path/'val'/img),str(target_path / 'val2'/ img))
    # print("All copied")
    # exit()

    data = (ObjectItemList.from_folder(path,valid='val')
            .split_by_rand_pct()
            .label_from_func(get_y_func)
            .transform(get_transforms(), tfm_y=True, size=imgsize)
            .databunch(bs=batch_size, collate_fn=bb_pad_collate).normalize(coco_stat))

    return data

def object365_tiny(batch_size=8, imgsize=300, **kwargs):
    path = Path('/datasets/object365_tiny')
    train_images, train_lbl_bbox = get_annotations(path / 'objects365_Tiny_train.json')
    val_images, val_lbl_bbox = get_annotations(path / 'objects365_Tiny_val.json')
    # test_images, test_lbl_bbox = get_annotations(path / 'objects365_Tiny_Testset_images_list.json')
    images, lbl_bbox = train_images + val_images, train_lbl_bbox + val_lbl_bbox
    img2bbox = dict(zip(images, lbl_bbox))
    get_y_func = lambda o: img2bbox[o.name]

    data = (ObjectItemList.from_folder(path)
            .split_by_rand_pct()
            .label_from_func(get_y_func)
            .transform(get_transforms(), tfm_y=True, size=imgsize)
            .databunch(bs=batch_size, collate_fn=bb_pad_collate).normalize(coco_stat))

    return data
