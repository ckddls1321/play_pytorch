import torch.optim as optim
import math
import torch

__all__ = ['get_optimizer']

def get_optimizer(config):
    if config['type']=="SGD":
        optimizer = partial(optim.SGD,momentum=config['momentum'])
    else:
        print("Error : Not Supported Type of Optimizer", config['type'],"Not Im,plemented Yet")
        exit()
    return optimizer
