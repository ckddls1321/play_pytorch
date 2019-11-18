from pathlib import Path
import copy
from fastai.vision import *
from fastai.metrics import *

def oxford_iiit_pet(batch_size,imgsize=224,**kwargs):
    # path = untar_data(URLs.PETS)
    path = url2path(URLs.PETS)
    pat = r'/([^/]+)_\d+.jpg$'
    fnames = get_image_files(path / 'images')
    dataset = ImageDataBunch.from_name_re(path, fnames, pat, ds_tfms=get_transforms(), size=224).normalize(imagenet_stats)
    return dataset

def get_annotations_from_txt(fname, prefix=None):
    images = []
    labels = []
    with open(str(fname), 'r') as f:
        while True:
            line = f.readline()
            if not line: break
            img, lbl = line.replace('\n','').split(' ')
            images.append(Path(img).name)
            labels.append(lbl)

    return images, labels

def oxford_102_flowers(batch_size,imgsize=224,**kwargs):
    # path = untar_data(URLs.FLOWERS)
    path = url2path(URLs.FLOWERS)
    train_images, train_lbl = get_annotations_from_txt(path / 'train.txt')
    val_images, val_lbl = get_annotations_from_txt(path / 'valid.txt')
    test_images, test_lbl = get_annotations_from_txt(path / 'test.txt')
    images, lbl = train_images + val_images, train_lbl + val_lbl
    img2label = dict(zip(images, lbl))
    get_y_func = lambda o: img2label[o.name]
    fnames = get_image_files(path / 'jpg')
    fnames_to_remove = [path / 'jpg' / img for img in test_images]
    for fname in fnames_to_remove:
        fnames.remove(fname)

    dataset = (ImageDataBunch.from_name_func(path,fnames, get_y_func, valid_pct=0.2, ds_tfms=get_transforms(), size=imgsize, bs=batch_size)
               # .add_test(test_images,tfms=([*rand_pad(10, imgsize), flip_lr(p=0.5)],[]),label=test_lbl)
               .normalize(imagenet_stats))


    return dataset
