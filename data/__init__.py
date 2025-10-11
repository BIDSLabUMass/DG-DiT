import torch.utils.data
from .myloader import *

def creat_trainloader(batch_size):
    return get_train_loaders(batch_size)

def creat_testloader(phase, scanner, tracer):
    if phase == 'test':
        return get_test_loader(scanner, tracer, 1)

    elif phase == 'PiB':
        return get_pib_loader()