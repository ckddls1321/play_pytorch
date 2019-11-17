from pathlib import Path
import copy
from fastai.vision import *
from fastai.metrics import *

def camvid(batch_size,imgsize=(360,480),**kwargs):
    path = untar_data(URLs.CAMVID)
    path_lbl = path / 'labels'
    path_img = path / 'images'
    codes = np.loadtxt(path / 'codes.txt', dtype=str)
    get_y_fn = lambda x: path_lbl / f'{x.stem}_P{x.suffix}'
    dataset = (SegmentationItemList.from_folder(path_img)
            .split_by_fname_file('../valid.txt')
            .label_from_func(get_y_fn, classes=codes)
            .transform(get_transforms(), size=imgsize, tfm_y=True)
            .databunch(bs=batch_size, num_workers=0)
            .normalize(imagenet_stats))
    return dataset
