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
import loss
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
    parser.add_argument('--dist_url', default='tcp://224.66.41.62:23456',
                        help='distributed url used to set up, host process')
    parser.add_argument('--dist_backend', default='nccl', help='distributed url used to set up, host process')
    parser.add_argument('--multiprocessing_distributed', default='nccl',
                        help='distributed url used to set up, host process')
    parser.add_argument('--save', default=False, help='Save after training')
    parser.add_argument('--arch_search', default=False, help='Neural Architecture Search')
    parser.add_argument('--quantize', default=False, help='Quantization Model')
    parser.add_argument('--prune', default=False, help='pruning')
    args = parser.parse_args()
    if args.multiprocessing_distributed:
        args.world_size = torch.cuda.device_count() * args.world_size
    if args.local_rank > -1:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend=args.dist_backend)
    return args


def set_env(nbatch):
    torch.cuda.empty_cache()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        ncuda = torch.cuda.device_count()
        nbatch = nbatch // ncuda
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
    _, device, _ = set_env(cfg.total_bs)

    try:
        model = ptcv_get_model(cfg.model.type, pretrained=cfg.model.pretrained)
    except:
        print("Model Not Implemented. ")
        exit()

    try:
        dataset = getattr(data, cfg.dataset_type)
        dataset = dataset(batch_size=cfg.total_bs, imgsize=cfg.img_size)
        dataset.show_batch(rows=3, figsize=(10,10))
        plt.show()
    except:
        print("Dataset Type {} are Not Implemented. ".format(cfg.dataset_type))
        exit()


    metrics = [getattr(fastai.metrics, met) for met in cfg.metric]
    try:
        loss = getattr(nn, cfg.loss)
    except AttributeError as error:
        loss = getattr(loss, cfg.loss)
    optimizer = get_optimizer(cfg.optimizer)

    # learner = Learner(dataset, model, opt_func=optimizer, loss_func=loss().to(device),
    #                   metrics=metrics, bn_wd=False, true_wd=True,
    #                   wd=cfg.optimizer.weight_decay, path=cfg.work_dir) # Custom Learner

    # models : Darknet, resnet18,34,50,101,152,xresnet18,34,50,101,152,squeezenet1_0,squeezenet1_1, densenet121
    # learner.loss_func :
    # learner = cnn_learner(dataset, models.resnet18, wd=cfg.optimizer.weight_decay)
    # learner = unet_learner(dataset, models.resnet34, metrics=partial(foreground_acc,void_code=30), wd=cfg.optimizer.weight_decay)
    # languagemodellearner()
    # textclassifierlearner()
    # tabular_learner()

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
        SaveModelCallback(learner, name='model_best', monitor='accuracy'),
        # LearnerTensorboardWriter(learner,name=Path('runs/'),base_dir=Path(cfg.work_dir))
    ]

    for callback in callbacks:
        learner.callbacks.append(callback)

    # learner.mixup()
    # learner.cutmix()
    # learner.ricap()
    # learner.blend()
    # learner.show_tfms()

    print(cfg.text)
    # learner.lr_find(num_it=200)
    # learner.recorder.plot(suggestion=True,show_moms=False)
    # learner.sched.plot_lr(show_moms=False)
    # exit()

    if cfg.lr_config.policy.lower() == 'cyclic':
        learner.fit_one_cycle(int(cfg.total_epochs), cfg.optimizer.lr, tot_epochs=int(cfg.total_epochs),
                              div_factor=cfg.lr_config.div_factor,pct_start=cfg.lr_config.pct_start)
    if cfg.lr_config.policy.lower() == 'cosine':
        fit_warmup_cosannealing(learner, cfg.optimizer.lr, cfg.total_epochs, warmup_ratio=cfg.lr_config.warmup_ratio)
    if cfg.lr_config.policy.lower() == 'step':
        fit_warmup_multistep(learner, cfg.optimizer.lr, cfg.lr_config.gamma, cfg.lr_config.step, cfg.total_epochs,
                             warmup_ratio=cfg.lr_config.warmup_ratio)
    # if cfg.lr_config.policy.lower() == 'warm_restart':
    #     fit_warmup_restart(learner,n_cycles, cfg.optimizer.lr, cfg.optimizer.mom, cycle_len, cycle_mult)

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
