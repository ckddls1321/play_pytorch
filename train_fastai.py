from pathlib import Path
import json
import argparse
import fastai
from fastai.vision import *
from fastai.metrics import *
from fastai.callbacks import *
from fastai.callbacks.tensorboard import LearnerTensorboardWriter
from fastai.vision.models import *
from fastai.distributed import *
from pytorchcv.model_provider import get_model as ptcv_get_model
from mmcv import Config as Cfg
from mmcv.runner import load_checkpoint
import data
from utils import *

def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work_dir', help='the dir to save logs and models')
    parser.add_argument('--resume_from', help='the checkpoint file to resume from')
    parser.add_argument('--validate', help='validate model')
    parser.add_argument('--world_size', default=-1, help='number of nodes for distributed training')
    parser.add_argument('--rank', default=-1, help='node rank for distributed training')
    parser.add_argument('--local_rank', default=-1, help='node rank for distributed training')
    parser.add_argument('--dist_url', default='tcp://224.66.41.62:23456', help='distributed url used to set up, host process')
    parser.add_argument('--dist_backend', default='nccl', help='distributed url used to set up, host process')
    parser.add_argument('--multiprocessing_distributed', default='nccl', help='distributed url used to set up, host process')
    parser.add_argument('--arch_search', default=False, help='Neural Architecture Search')
    parser.add_argument('--quantize', default=False, help='Quantization Model')
    parser.add_argument('--prune', default=False, help='pruning')
    args = parser.parse_args()
    if args.multiprocessing_distributed:
        args.world_size = ngpus_per_node * args.world_size
    if args.local_rank > -1:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend=args.dist_backend)
    return args

def set_env(nbatch):
    torch.cuda.empty_cache()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        ncuda = torch.cuda.device_count()
        nbatch = nbatch //ncuda
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True
        torch.manual_seed(1)
        torch.cuda.manual_seed(1)
    else:
        ncuda = 0
    return nbatch, device, ncuda


if __name__ == "__main__":
    gc.collect()
    args = parse_args()
    cfg = Cfg.fromfile(args.config)
    set_env(cfg.total_bs)

    try:
        model = ptcv_get_model(cfg.model.type, pretrained=cfg.model.pretrained)
    except:
        print("Model Not Implemented. ")
        exit()

    try:
        dataset = getattr(data, cfg.dataset_type)
        dataset = dataset(cfg.data_root, batch_size = cfg.total_bs, imgsize=cfg.img_size)
    except:
        print("Dataset Type {} are Not Implemented. ")
        exit()

    metrics = [getattr(fastai.metrics,met) for met in cfg.metric]
    loss = getattr(nn, cfg.loss)
    # try :
    #     loss = getattr(nn,cfg.loss)
    # except AttributeError as error:
    #     loss = getattr(loss, cfg.loss)
    optimizer = get_optimizer(cfg.optimizer)

    learner = Learner(dataset, model, opt_func=optimizer, loss_func=loss().to(device), metrics=metrics, bn_wd=False, true_wd=True, wd=cfg.optimizer.weight_decay,path=cfg.work_dir)

    if args.local_rank > -1:
        learner.to_distributed(args.local_rank)
    else:
        learner.to_parallel()

    if args.validate:
        learner.model.eval()
        print(learner.validate())
        exit()

    learner.to_fp16()

    callbacks = [
        SaveModelCallback(learner,name='model_best',monitor='accuracy')
        LearnerTensorboardWriter(learner,name=Path('runs/'),base_dir=Path(cfg.work_dir))
    ]

    for callback in callbacks:
        learner.callbacks.append(callback)

    learner.mixup()
    learner.cutmix()
    learner.ricap()
    learner.blend()
    learner.show_tfms()

    print(cfg.text)
    # learner.lr_find(num_it=50, wd=cfg.optimizer.wd)
    # learner.recorder.plot_lr(suggestion=True,show_moms=False)
    if cfg.lr_config.policy.lower() == 'cyclic':
        learner.fit_one_cycle(int(cfg.total_epochs),cfg.optimizer.lr,moms=(0.95,0.8), tot_epochs=int(cfg.total_epochs))
    if cfg.lr_config.policy.lower() == 'cosine':
        fit_warmup_cosannealing(learner,cfg.optimizer.lr, 0.9, cfg.total_epochs, warmup_ratio=cfg.lr_config.warmup_ratio)
    if cfg.lr_config.policy.lower() == 'step':
        fit_warmup_cosannealing(learner,cfg.optimizer.lr, cfg.lr_config.gamma, cfg.lr_config.step, cfg.total_epochs, warmup_ratio=cfg.lr_config.warmup_ratio)

    # Interpretation
    # learner.to_fp32()
    # learner.model.eval()
    # interp = ClassificationInterpretation.from_learner(learner, DatasetType.Valid, tta=True)
    # interp.plot_top_losses(16, figsize=(10,10))
    # interp.plot_confusion_matrix(slice_size=10)
    # interp.most_confused(min_val=2, slice_size=10)
    # tta_params = {'beta':0.12, 'scale':1.0}
    # preds, y = learner.TTA(ds_type=DatasetType.Valid,**tta_params)
    # preds.argmax(1)

    plt.show()