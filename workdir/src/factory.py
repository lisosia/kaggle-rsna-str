from torch.utils.data import DataLoader
import torch.optim
from torch.optim import lr_scheduler

import src.models as models
import src.criterion as criterion
import src.datasets as datasets


def get_model(config):
    c = config["model"]
    return getattr(models, c["class"])(**c["param"])


def get_criterion(config):
    c = config["criterion"]
    return getattr(criterion, c["class"])(**c["param"])


def get_scheduler(config, optimizer: torch.optim.Optimizer):
    c = config["scheduler"]
    sched = getattr(lr_scheduler, c["class"])(optimizer, **c["param"])
    is_reduce_lr = isinstance(sched, lr_scheduler.ReduceLROnPlateau)
    return sched, is_reduce_lr


def get_optimizer(config, model_parameters):
    c = config["optimizer"]
    optimizer = getattr(torch.optim, c["class"])(model_parameters, **c["param"])
    return optimizer


def get_loader_train(config):
    c = config["dataset"]
    dataset_train = getattr(datasets, c["class"])(config["fold"], "train", **c["param"])
    c = config["dataloader_train"]
    loader_train = DataLoader(dataset_train, shuffle=True, pin_memory=True, **c["param"])  #bs=56 for effnet
    return loader_train

def get_loader_valid(config):
    c = config["dataset"]
    dataset_valid = getattr(datasets, c["class"])(config["fold"], "valid", **c["param"])
    c = config["dataloader_valid"]
    loader_train = DataLoader(dataset_valid, shuffle=False, pin_memory=True, **c["param"])  #bs=96 for effnet
    return loader_train
