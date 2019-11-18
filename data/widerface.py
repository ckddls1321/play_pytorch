from pathlib import Path
import copy
from fastai.vision import *
from fastai.datasets import *
from fastai.metrics import *
import collections

__all__ = ['widerface']

wider_stat = ([0.483, 0.454, 0.404], [1,1,1])

def get_annotations_wider_txt(fname, prefix=None):
    id2images, id2bboxes, id2cats = {}, collections.defaultdict(list), collections.defaultdict(list)
    img_id = 0
    cat_id = 1
    categories = [{"id": 1, "name": 'face'}]
    with open(str(fname), 'r') as f:
        while True:
            image_file = f.readline()
            if not image_file:
                break
            image_file = image_file.replace('\n','')
            image = {}
            image['id'] = img_id
            img_id +=1
            image['file_name'] = Path(image_file).name
            nBBox = f.readline()
            i = 0
            while i < int(nBBox):
                x1, y1, w, h, _, _, _, _, _, _ = [int(i) for i in f.readline().split()]
                if w > 0 and h > 0:
                    id2bboxes[image['id']].append([y1, x1, y1+h, x1+w])
                    id2cats[image['id']].append('face')
                i = i + 1
            if int(nBBox) == 0:
                x1, y1, w, h, _, _, _, _, _, _ = [int(i) for i in f.readline().split()]
            id2images[image['id']] = ifnone(prefix, '') + image['file_name']
    ids = list(id2images.keys())
    return [id2images[k] for k in ids], [[id2bboxes[k], id2cats[k]] for k in ids]


def widerface(data_root=Path.home() / 'DB', batch_size=32, imgsize=256, **kwargs):
    wider_path = data_root / 'widerface/'
    train_images, train_lbl_bbox = get_annotations_wider_txt(wider_path / 'wider_face_split/wider_face_train_bbx_gt.txt')
    val_images, val_lbl_bbox = get_annotations_wider_txt(wider_path / 'wider_face_split/wider_face_val_bbx_gt.txt')
    images, lbl_bbox = train_images + val_images, train_lbl_bbox + val_lbl_bbox
    img2bbox = dict(zip(images, lbl_bbox))
    get_y_func = lambda o: img2bbox[o.name]
    get_valid_func = lambda o: o.name in val_images
    
    fnames = get_image_files(wider_path/ 'train',recurse=False)

    data = (ObjectItemList.from_folder(wider_path / 'train')
            .split_by_files(val_images)
            .label_from_func(get_y_func)
            .transform(get_transforms(), tfm_y=True, size=imgsize)
            .databunch(bs=batch_size, collate_fn=bb_pad_collate).normalize(wider_stat))

    return data
