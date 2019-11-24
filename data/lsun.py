from pathlib import Path
from fastai.vision import *
from fastai.vision.gan import *
from fastai.metrics import *
from matplotlib import pyplot as plt

def lsun(batch_size=128,imgsize=64, **kwargs):
    # path = untar_data(URLs.LSUN_BEDROOMS)
    path = url2path(URLs.LSUN_BEDROOMS)
    print(path)
    dataset = (GANItemList.from_folder(path, noise_sz=100)
     .no_split()
     .label_from_func(noop)
     .transform(tfms=[[crop_pad(size=imgsize, row_pct=(0, 1), col_pct=(0, 1))], []], size=imgsize, tfm_y=True)
     .databunch(bs=batch_size)
     .normalize(stats=[torch.tensor([0.5, 0.5, 0.5]), torch.tensor([0.5, 0.5, 0.5])], do_x=False, do_y=True))
    return dataset

