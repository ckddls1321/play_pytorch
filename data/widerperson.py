from pathlib import Path
import copy
from fastai.vision import *
from fastai.datasets import *
from fastai.metrics import *
import collections

__all__ = ['widerperson']

wider_stat = ([0.483, 0.454, 0.404], [1,1,1])

categories = ['pedestrians','riders','partially-visible persons','ignore regions','crowd']

def get_annotations_wider_txt(fname, prefix=None):
    id2images, id2bboxes, id2cats = {}, collections.defaultdict(list), collections.defaultdict(list)
    img_id = 0
    categories = [{"id": 1, "name": 'pedestrians'},{"id": 2, "name": 'riders'},
                  {"id": 3, "name": 'partial-visible-person'},{"id":4, "name": 'crowd'},
                  {"id": 5, "name": 'ignore-region'}
                  ]
    with open(str(fname), 'r') as f:
        classes = {}
        for o in categories:
            classes[o['id']] = o['name']
        while True:
            image_file = f.readline()
            if not image_file:
                break
            image_file = image_file.replace('\n','')
            image = {}
            image['id'] = img_id
            img_id +=1
            image['file_name'] = image_file+'.jpg'
            with open(str(Path(fname).parent) + '/Annotations/' + image_file+'.jpg.txt','r') as bbox_annotation_f:
                nBBox = bbox_annotation_f.readline()
                i = 0
                while i < int(nBBox):
                    id, x1, y1, x2, y2 = [int(i) for i in bbox_annotation_f.readline().split()]
                    if (x2-x1) > 0 and (y2-y1) > 0:
                        id2bboxes[image['id']].append([y1, x1, y2, x2])
                        id2cats[image['id']].append(classes[id])
                    i = i + 1
            id2images[image['id']] = ifnone(prefix, '') + image['file_name']
    ids = list(id2images.keys())
    return [id2images[k] for k in ids], [[id2bboxes[k], id2cats[k]] for k in ids]

def widerperson(batch_size=32, imgsize=256, **kwargs):
    wider_path = Path('/datasets/widerperson/')
    train_images, train_lbl_bbox = get_annotations_wider_txt(str(wider_path / 'train.txt'))
    val_images, val_lbl_bbox = get_annotations_wider_txt(str(wider_path / 'val.txt'))
    # test_images, test_lbl_bbox = get_annotations_wider_txt(str(wider_path / 'test.txt'))
    images, lbl_bbox = train_images + val_images, train_lbl_bbox + val_lbl_bbox
    img2bbox = dict(zip(images, lbl_bbox))
    get_y_func = lambda o: img2bbox[o.name]

    # import shutil
    # for img in train_images:
    #     shutil.copy(str(wider_path/'Images'/img),str(wider_path / 'train'/ img))
    # for img in val_images:
    #     shutil.copy(str(wider_path/'Images'/img),str(wider_path / 'train'/ img))
    # print("All copied")
    # exit()

    data = (ObjectItemList.from_folder(wider_path / 'train')
            .split_by_files(val_images)
            .label_from_func(get_y_func)
            .transform(get_transforms(), tfm_y=True, size=imgsize)
            .databunch(bs=batch_size, collate_fn=bb_pad_collate).normalize(wider_stat))

    return data
