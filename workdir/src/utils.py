import codecs
import json
import logging
import os
import random
import time

import numpy as np
import torch
import yaml
import pickle

from contextlib import contextmanager
from typing import Union, Optional
from pathlib import Path


@contextmanager
def timer(name: str, logger: Optional[logging.Logger] = None):
    t0 = time.time()
    msg = f"[{name}] start"
    if logger is None:
        print(msg)
    else:
        logger.info(msg)
    yield

    msg = f"[{name}] done in {time.time() - t0:.2f} s"
    if logger is None:
        print(msg)
    else:
        logger.info(msg)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = True  # type: ignore


class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)

def save_json(config: dict, save_path: Union[str, Path]):
    f = codecs.open(str(save_path), mode="w", encoding="utf-8")
    json.dump(config, f, indent=4, cls=MyEncoder, ensure_ascii=False)
    f.close()


def save_pickle(obj, path: os.PathLike):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(obj, f)

def load_pickle(path: os.PathLike):
    with open(path, 'rb') as f:
        obj = pickle.load(obj, f)
    return obj


def load_config(path: str):
    with open(path) as f:
        config = yaml.safe_load(f)
    return config


def save_model(model, optim, detail, fold, dirname):
    path = os.path.join(dirname, 'fold%d_ep%d.pt' % (fold, detail['epoch']))
    torch.save({
        'model': model.state_dict(),
        'optim': optim.state_dict(),
        'detail': detail,
    }, path)
    print('saved model to %s' % path)


def load_model(path, model, optim=None):

    # remap everthing onto CPU 
    state = torch.load(str(path), map_location=lambda storage, location: storage)

    model.load_state_dict(state['model'])
    if optim:
        print('loading optim too')
        optim.load_state_dict(state['optim'])
    else:
        print('not loading optim')

    model.cuda()

    detail = state['detail']
    print('loaded model from %s' % path)

    return detail


def get_lr(optim):
    if optim:
        return optim.param_groups[0]['lr']
    else:
        return 0


def get_logger(out_file=None):
    logger = logging.getLogger()
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    logger.handlers = []
    logger.setLevel(logging.INFO)

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)

    if out_file is not None:
        fh = logging.FileHandler(out_file)
        fh.setFormatter(formatter)
        fh.setLevel(logging.INFO)
        logger.addHandler(fh)
    logger.info("logger set up")
    return logger
