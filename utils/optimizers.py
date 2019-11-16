import torch.optim as optim
import math
import torch
from functools import partial

__all__ = ['get_optimizer']

def get_optimizer(config):
    if config['type'].lower()=="sgd":
        optimizer = partial(optim.SGD,momentum=config['momentum'])
    elif config['type'].lower() == "adamw":
        optimizer = partial(optim.AdamW, betas=config['betas'],eps=config['eps'])
    else:
        print("Error : Not Supported Type of Optimizer", config['type'],"Not Im,plemented Yet")
        exit()
    return optimizer
