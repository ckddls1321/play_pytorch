from pathlib import Path
import copy
from fastai.vision import *
from fastai.metrics import *

def food101(batch_size=256, imgsize=224, **kwargs):
    # path = untar_data(URLs.FOOD)
    path = url2path(URLs.FOOD)
    dataset = (ImageDataBunch.from_folder(path,valid_pct=0.2,test='test',ds_tfms=get_transforms(),size=imgsize,bs=batch_size,**kwargs).normalize(imagenet_stats))
    return dataset

def fruits360(batch_size=256, imgsize=224, **kwargs):
    path = Path('/datasets/fruits-360/')
    dataset = (ImageDataBunch.from_folder(path,train='Training',valid_pct=0.2,test='Test',ds_tfms=get_transforms(),size=imgsize,bs=batch_size,**kwargs).normalize(imagenet_stats))
    return dataset

# refer tohttps://github.com/marcusklasson/GroceryStoreDataset
def save_grocery_txt_to_csv(path, prefix=None):
    categories = [
        "Golden - Delicious","Granny - Smith","Pink - Lady","Red - Delicious",
        "Royal - Gala","Avocado","Banana","Kiwi","Lemon","Lime","Mango","Cantaloupe",
        "Galia - Melon","Honeydew - Melon","Watermelon","Nectarine","Orange","Papaya",
        "Passion - Fruit","Peach","Anjou","Conference","Kaiser","Pineapple",
        "Plum","Pomegranate","Red - Grapefruit","Satsumas","Bravo - Apple - Juice",
        "Bravo - Orange - Juice","God - Morgon - Apple - Juice",
        "God - Morgon - Orange - Juice","God - Morgon - Orange - Red - Grapefruit - Juice",
        "God - Morgon - Red - Grapefruit - Juice","Tropicana - Apple - Juice",
        "Tropicana - Golden - Grapefruit","Tropicana - Juice - Smooth","Tropicana - Mandarin - Morning",
        "Arla - Ecological - Medium - Fat - Milk","Arla - Lactose - Medium - Fat - Milk",
        "Arla - Medium - Fat - Milk","Arla - Standard - Milk",
        "Garant - Ecological - Medium - Fat - Milk","Garant - Ecological - Standard - Milk",
        "Oatly - Natural - Oatghurt","Oatly - Oat - Milk",
        "Arla - Ecological - Sour - Cream","Arla - Sour - Cream","Arla - Sour - Milk",
        "Alpro - Blueberry - Soyghurt","Alpro - Vanilla - Soyghurt",
        "Alpro - Fresh - Soy - Milk","Alpro - Shelf - Soy - Milk",
        "Arla - Mild - Vanilla - Yoghurt","Arla - Natural - Mild - Low - Fat - Yoghurt",
        "Arla - Natural - Yoghurt","Valio - Vanilla - Yoghurt",
        "Yoggi - Strawberry - Yoghurt","Yoggi - Vanilla - Yoghurt",
        "Asparagus","Aubergine","Cabbage","Carrots","Cucumber","Garlic","Ginger",
        "Leek","Brown - Cap - Mushroom","Yellow - Onion",
        "Green - Bell - Pepper","Orange - Bell - Pepper",
        "Red - Bell - Pepper","Yellow - Bell - Pepper","Floury - Potato",
        "Solid - Potato","Sweet - Potato","Red - Beet","Beef - Tomato",
        "Regular - Tomato","Vine - Tomato","Zucchini"
    ]
    with open(str(path / 'train.csv'), 'w') as out_file:
        with open(str(path / 'train.txt'), 'r') as in_file:
            stripped = (line.strip() for line in in_file)
            lines = (line.split(",") for line in stripped if line)
            writer = csv.writer(out_file)
            writer.writerows(lines)
        with open(str(path / 'val.txt'), 'r') as in_file:
            stripped = (line.strip() for line in in_file)
            lines = (line.split(",") for line in stripped if line)
            writer = csv.writer(out_file)
            writer.writerows(lines)
        with open(str(path / 'test.txt'), 'r') as in_file:
            stripped = (line.strip() for line in in_file)
            lines = (line.split(",") for line in stripped if line)
            writer = csv.writer(out_file)
            writer.writerows(lines)

def grocerystore(batch_size=256, imgsize=224, **kwargs):
    # Image File Name Conflict
    path = Path('/datasets/GroceryStoreDataset/dataset/')
    save_grocery_txt_to_csv(path)
    dataset = ImageDataBunch.from_csv(path,csv_labels='train.csv',label_delim=',',valid_pct=0.2,ds_tfms=get_transforms(),size=imgsize,bs=batch_size).normalize(imagenet_stats)
    return dataset

