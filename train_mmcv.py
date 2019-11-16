import logging
import os
from argparse import ArgumentParser
from collections import OrderedDict

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.nn.parallel import DataParallel, DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms

from mmcv import Config
from mmcv.runner import DistSamplerSeedHook, Runner
from fastai.metrics import *
from pytorchcv.model_provider import get_model as ptcv_get_model

import data

def batch_processor(model, data, train_mode):
    img, label = data
    label = label.cuda(non_blocking=True)
    pred = model(img)
    loss = F.cross_entropy(pred, label)
    acc_top1 = top_k_accuracy(pred, label, 1)
    acc_top5 = top_k_accuracy(pred, label, 5)
    log_vars = OrderedDict()
    log_vars['loss'] = loss.item()
    log_vars['acc_top1'] = acc_top1.item()
    log_vars['acc_top5'] = acc_top5.item()
    outputs = dict(loss=loss, log_vars=log_vars, num_samples=img.size(0))
    return outputs

def get_logger(log_level):
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=log_level)
    logger = logging.getLogger()
    return logger

def init_dist(backend='nccl', **kwargs):
    if mp.get_start_method(allow_none=True) is None:
        mp.set_start_method('spawn')
    rank = int(os.environ['RANK'])
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(rank % num_gpus)
    dist.init_process_group(backend=backend, **kwargs)

def parse_args():
    parser = ArgumentParser(description='Train CIFAR-10 classification')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--launcher',choices=['none', 'pytorch'],default='none',help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    return parser.parse_args()

def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    logger = get_logger(cfg.log_level)

    # init distributed environment if necessary
    if args.launcher == 'none':
        dist = False
        logger.info('Disabled distributed training.')
    else:
        dist = True
        init_dist(**cfg.dist_params)
        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()
        if rank != 0:
            logger.setLevel('ERROR')
        logger.info('Enabled distributed training.')

    try:
        dataset = getattr(data, cfg.dataset_type)
        train_loader, val_loader = dataset(cfg.data_root, batch_size = cfg.total_bs, imgsize=cfg.img_size)
    except:
        print("Dataset Type {} are Not Implemented. ")
        exit()

    # build model
    try:
        model = ptcv_get_model(cfg.model.type, pretrained=cfg.model.pretrained)
    except:
        print("Model Not Implemented. ")
        exit()

    if dist:
        model = DistributedDataParallel(model.cuda(), device_ids=[torch.cuda.current_device()])
    else:
        model = DataParallel(model, device_ids=cfg.gpus).cuda()

    # build runner and register hooks
    runner = Runner(model,batch_processor,cfg.optimizer,cfg.work_dir,log_level=cfg.log_level)
    runner.register_training_hooks(
        lr_config=cfg.lr_config,
        optimizer_config=cfg.optimizer_config,
        checkpoint_config=cfg.checkpoint_config,
        log_config=cfg.log_config)
    if dist:
        runner.register_hook(DistSamplerSeedHook())

    # load param (if necessary) and run
    if cfg.get('resume_from') is not None:
        runner.resume(cfg.resume_from)
    elif cfg.get('load_from') is not None:
        runner.load_checkpoint(cfg.load_from)

    runner.run([train_loader, val_loader], cfg.workflow, cfg.total_epochs)


if __name__ == '__main__':
    main()