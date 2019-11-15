total_epochs = 174
total_bs = 512
img_size = 32
log_level = 'INFO'
load_from = None
resume_from = None
dataset_type = 'cifar10'
model = dict(
    type='vgg7_cifar10',
    pretrained=False
)
work_dir = './work_dirs/' + dataset_type + '_' + model['type']
data_root = '/home/ckddls1321/'
loss = "CrossEntropyLoss"
metric = ["accuracy","error_rate"]
optimizer = dict(type='SGD',lr=0.1, momentum=0.9, weight_decay=1e-4)
lr_config = dict(policy='cosine',iter=total_epochs,warmup='linear', warmup_iters=3900, warmup_ratio=1.0)
