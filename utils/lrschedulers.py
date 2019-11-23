import math
import torch
import torch.optim.lr_scheduler
from functools import partial

__all__ = ['get_lrscheduler']

def get_lrscheduler(config, optimizer):
    scheduler = getattr(torch.optim.lr_scheduler, config['type'])
    if config['type'].lower() == 'steplr':
        scheduler = scheduler(optimizer,step_size=config['step_size'],gamma=config['gamma'],last_epoch=-1)
    return scheduler

if __name__ == '__main__':
    net = torch.nn.Conv2d(3,16,3)
    optimizer = torch.optim.SGD(net.parameters(),lr=0.01)
    config = {'type':'StepLR','step_size':5,'gamma':0.8}
    get_lrscheduler(config,optimizer)
