from pathlib import Path
import copy
import numpy as np
from fastai.vision import *
from fastai.metrics import *



def biwihead(batch_size,imgsize=(120,160), **kwargs):
    path = url2path(URLs.BIWI_HEAD_POSE)
    cal = np.genfromtxt(path / '01' / 'rgb.cal', skip_footer=6)
    def img2txt_name(f): return path / f'{str(f)[:-7]}pose.txt'
    def convert_biwi(coords):
        c1 = coords[0] * cal[0][0] / coords[2] + cal[0][2]
        c2 = coords[1] * cal[1][1] / coords[2] + cal[1][2]
        return tensor([c2, c1])
    def get_ctr(f):
        ctr = np.genfromtxt(img2txt_name(f), skip_header=3)
        return convert_biwi(ctr)
    def get_ip(img, pts): return ImagePoints(FlowField(img.size, pts), scale=True)
    dataset = (PointsItemList.from_folder(path)
            .split_by_valid_func(lambda o: o.parent.name == '13')
            .label_from_func(get_ctr)
            .transform(get_transforms(), tfm_y=True, size=imgsize) # default : 1
            .databunch(bs=batch_size, num_workers=0).normalize(imagenet_stats)
            )
    return dataset
