from fastai.callbacks import *
from fastai.callback import *

def fit_warmup_restart(learn, n_cycles, lr, mom, cycle_len, cycle_mult):
    n = len(learn.data.train_dl)
    phases = [(TrainingPhase(n * (cycle_len * cycle_mult**i))
                 .schedule_hp('lr', lr, anneal=annealing_cos)
                 .schedule_hp('mom', mom)) for i in range(n_cycles)]
    sched = GeneralScheduler(learn, phases)
    learn.callbacks.append(sched)
    if cycle_mult != 1:
        total_epochs = int(cycle_len * (1 - (cycle_mult)**n_cycles)/(1-cycle_mult))
    else: total_epochs = n_cycles * cycle_len
    learn.fit(total_epochs)

def fit_warmup_cosannealing(learn, lr, total_epochs, warmup_ratio=0.1):
    n = len(learn.data.train_dl)
    phases = [
        (TrainingPhase(total_epochs * warmup_ratio * n).schedule_hp('lr', (lr / 1e4, lr), anneal=annealing_linear)),
        (TrainingPhase(n * total_epochs * (1-warmup_ratio)).schedule_hp('lr', lr, anneal=annealing_cos))
    ]
    sched = GeneralScheduler(learn, phases)
    learn.callbacks.append(sched)
    learn.fit(total_epochs)


def fit_warmup_multistep(learn, lr, gamma, step, total_epochs, warmup_ratio=0.1):
    n = len(learn.data.train_dl)
