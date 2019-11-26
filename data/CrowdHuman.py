from pathlib import Path
import copy
from fastai.vision import *
from fastai.metrics import *
import json
import numpy as np

def get_annotations_odgt(fname, prefix=None):
    "Open a COCO style json in `fname` and returns the lists of filenames (with maybe `prefix`) and labelled bboxes."
    id2images, id2bboxes, id2cats = {}, collections.defaultdict(list), collections.defaultdict(list)
    with open(fname, 'r+') as f:
        datalist = f.readlines()

    img_id = 0
    for i in np.arange(len(datalist)):
        adata = json.loads(datalist[i])
        gtboxes = adata['gtboxes']
        image = {}
        image['id'] = img_id
        img_id += 1
        image['file_name'] = adata['ID'] + '.jpg'
        for gtbox in gtboxes:
            if gtbox['tag'] == 'person':
                # Currently use only visible bbox
                # bbox = gtbox['fbox']
                # bbox[2] = bbox[2] + bbox[0]
                # bbox[3] = bbox[3] + bbox[1]
                # id2bboxes[image['id']].append(bbox)
                # id2cats[image['id']].append('person')
                # should consider mask and head occlusion etc
                bbox = gtbox['vbox'] # x, y, w, h
                bbox[2] = bbox[2] + bbox[0] # x1 + w
                bbox[3] = bbox[3] + bbox[1] # y1 + h
                id2bboxes[image['id']].append([bbox[1],bbox[0],bbox[3],bbox[2]]) # y1, x1, y2, x2
                id2cats[image['id']].append('person')
                hbox = gtbox['hbox']
                hbox[2] = hbox[2] + hbox[0]
                hbox[3] = hbox[3] + hbox[1]
                id2bboxes[image['id']].append([hbox[1],hbox[0],hbox[3],hbox[2]]) # y1, x1, y2, x2
                id2cats[image['id']].append('head')
            elif gtbox['tag'] == 'mask':
                pass
            else:
                print("??")
            id2images[image['id']] = ifnone(prefix, '') + image['file_name']
    ids = list(id2images.keys())
    return [id2images[k] for k in ids], [[id2bboxes[k], id2cats[k]] for k in ids]

def crowdhuman(batch_size=8, imgsize=300, **kwargs):
    path = Path('/datasets/CrowdHuman')
    # using jq, json command line parser
    train_images, train_lbl_bbox = get_annotations_odgt(str(path / 'annotation_train.odgt'))
    val_images, val_lbl_bbox = get_annotations_odgt(str(path / 'annotation_val.odgt'))
    images, lbl_bbox = train_images + val_images, train_lbl_bbox + val_lbl_bbox
    img2bbox = dict(zip(images, lbl_bbox))
    get_y_func = lambda o: img2bbox[o.name]

    data = (ObjectItemList.from_folder(path / 'Images')
            .split_by_files(val_images)
            .label_from_func(get_y_func)
            .transform(get_transforms(), tfm_y=True, size=imgsize)
            .databunch(bs=batch_size, collate_fn=bb_pad_collate).normalize(imagenet_stats))

    return data

