total_epochs = 70
total_bs = 256
img_size = 32
log_level = 'INFO'
load_from = None
resume_from = None
dataset_type = 'cifar10'
model = dict(
    type='seresnet20_cifar10',
    pretrained=False
)
work_dir = './work_dirs/' + dataset_type + '_' + model['type']
loss = "CrossEntropyLoss"
metric = ["accuracy","error_rate"]
# optimizer = dict(type='SGD',lr=0.1, momentum=0.9, weight_decay=1e-4)
optimizer = dict(type='AdamW',lr=3e-3, betas=(0.9,0.99), eps=1e-8, weight_decay=0.4)
# lr_config = dict(policy='cosine',iter=total_epochs,warmup_ratio=0.1)
lr_config = dict(policy='cyclic',iter=total_epochs,warmup_ratio=0.1,div_factor=10,pct_start=0.5)

