import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from sklearn import model_selection as sms

import src.datasets as datasets

from pathlib import Path
import glob

from src.criterion import ImgLoss  # noqa
# from src.transforms import *


def get_optimizer(model: nn.Module, config: dict):
    optimizer_config = config["optimizer"]
    optimizer_name = optimizer_config.get("name")

    return optim.__getattribute__(optimizer_name)(model.parameters(),
                                                  **optimizer_config["params"])


def get_scheduler(optimizer, config: dict):
    scheduler_config = config["scheduler"]
    scheduler_name = scheduler_config.get("name")

    if scheduler_name is None:
        return
    else:
        return optim.lr_scheduler.__getattribute__(scheduler_name)(
            optimizer, **scheduler_config["params"])


def get_criterion(config: dict):
    loss_config = config["loss"]
    loss_name = loss_config["name"]
    loss_params = {} if loss_config.get("params") is None else loss_config.get(
        "params")

    if hasattr(nn, loss_name):
        criterion = nn.__getattribute__(loss_name)(**loss_params)
    else:
        criterion_cls = globals().get(loss_name)
        if criterion_cls is not None:
            criterion = criterion_cls(**loss_params)
        else:
            raise NotImplementedError

    return criterion


def get_split(config: dict):
    split_config = config["split"]
    name = split_config["name"]

    return sms.__getattribute__(name)(**split_config["params"])


def get_metadata(config: dict):
    data_config = config["data"]
    with open(data_config["train_skip"]) as f:
        skip_rows = f.readlines()

    train = pd.read_csv(data_config["train_df_path"])
    audio_path = Path(data_config["train_audio_path"])

    for row in skip_rows:
        row = row.replace("\n", "")
        ebird_code = row.split("/")[1]
        filename = row.split("/")[2]
        train = train[~((train["ebird_code"] == ebird_code) &
                        (train["filename"] == filename))]
        train = train.reset_index(drop=True)
    return train, audio_path


def get_loader(fold: int,
               config: dict,
               phase: str):
    assert phase in ['train', 'valid']
    dataset_config = config["dataset"]
    dataset = datasets.RsnaDataset(fold=fold, phase=phase, **dataset_config)
    loader_config = config["loader"][phase]
    loader = data.DataLoader(dataset, **loader_config)
    return loader