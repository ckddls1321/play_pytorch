# Installing IceVision
# !pip install icevision[all] icedata

# Imports
from icevision.all import *
import icedata

# Load the PETS dataset
path = icedata.pets.load_data()

# Get the class_map, a utility that maps from number IDs to classs names
class_map = icedata.pets.class_map()



# PETS parser: provided out-of-the-box
parser = icedata.pets.parser(data_dir=path, class_map=class_map)
train_records, valid_records = parser.parse(data_splitter)

# shows images with corresponding labels and boxes
show_records(train_records[:6], ncols=3, class_map=class_map, show=True)

# Define transforms - using Albumentations transforms out of the box
train_tfms = tfms.A.Adapter(
    [*tfms.A.aug_tfms(size=384, presize=512), tfms.A.Normalize()]
)
valid_tfms = tfms.A.Adapter([*tfms.A.resize_and_pad(size), tfms.A.Normalize()])
# Create both training and validation datasets
train_ds = Dataset(train_records, train_tfms)
valid_ds = Dataset(valid_records, valid_tfms)

# Create both training and validation dataloaders
train_dl = faster_rcnn.train_dl(train_ds, batch_size=16, num_workers=4, shuffle=True)
valid_dl = faster_rcnn.valid_dl(valid_ds, batch_size=16, num_workers=4, shuffle=False)

# Create model
model = faster_rcnn.model(num_classes=len(class_map))

# Define metrics
metrics = [COCOMetric(metric_type=COCOMetricType.bbox)]

# Train using fastai2
learn = faster_rcnn.fastai.learner(
    dls=[train_dl, valid_dl], model=model, metrics=metrics
)
learn.fine_tune(10, lr=1e-4)





# COCO Example 
# pip install icevision[all] icedata

from icevision.all import *

url = "https://cvbp.blob.core.windows.net/public/datasets/object_detection/odFridgeObjects.zip"
dest_dir = "fridge"

# Loading Data
data_dir = icedata.load_data(url, dest_dir)

# Parser
class_map = ClassMap(["milk_bottle", "carton", "can", "water_bottle"])
parser = parsers.voc(annotations_dir=data_dir / "odFridgeObjects/annotations/",
                    images_dir=data_dir / "odFridgeObjects/images",
                    class_map=class_map)

# Records
train_records, valid_records = parser.parse()

# Transforms
train_tfms = tfms.A.Adapter([*tfms.A.aug_tfms(size=384, presize=512), tfms.A.Normalize()])
valid_tfms = tfms.A.Adapter([*tfms.A.resize_and_pad(384), tfms.A.Normalize()])

# Datasets
train_ds = Dataset(train_records, train_tfms)
valid_ds = Dataset(valid_records, valid_tfms)

# DataLoaders
train_dl = efficientdet.train_dl(train_ds, batch_size=16, num_workers=4, shuffle=True)
valid_dl = efficientdet.valid_dl(valid_ds, batch_size=16, num_workers=4, shuffle=False)

# Model and Metrics
model = efficientdet.model(model_name="tf_efficientdet_lite0", num_classes=len(class_map), img_size=size)
metrics = [COCOMetric(metric_type=COCOMetricType.bbox)]

# Training using Fastai
learn = efficientdet.fastai.learner(dls=[train_dl, valid_dl], model=model, metrics=metrics)
learn.fine_tune(50, 1e-2, freeze_epochs=20)

# Inference
# DataLoader
infer_dl = efficientdet.infer_dl(valid_ds, batch_size=8)
# Predict
samples, preds = efficientdet.predict_dl(model, infer_dl)
# Show samples
imgs = [sample["img"] for sample in samples]
show_preds(
    imgs=imgs[:6],
    preds=preds[:6],
    class_map=class_map,
    denormalize_fn=denormalize_imagenet,
    ncols=3,
)
