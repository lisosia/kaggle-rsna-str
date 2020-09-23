#!/usr/bin/env python3

import argparse
from pathlib import Path
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.optim
from torch.utils.data import DataLoader

import src.configuration as C
from src.models import get_img_model
import src.utils as utils
from src.utils import get_logger
from src.criterion import ImgLoss
from src.datasets import RsnaDataset

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=['train', 'valid'], help="train valid")
    parser.add_argument("config", help="Config file path")
    parser.add_argument("--fold", type=int, default=0, help="fold")
    parser.add_argument("--apex", action='store_true', default=False, help="apex")
    parser.add_argument("--output", "-o", help="output path for validation")
    parser.add_argument("--snapshot", "-s", help="snapshot weight path")
    return parser.parse_args()
args = get_args()
if args.apex:
    from apex import amp

EXP_ID = os.path.splitext(os.path.basename(args.config))[0]
SEED = 42
DEVICE = "cuda"

output_dir = Path("./output") / EXP_ID
output_dir.mkdir(exist_ok=True, parents=True)
_logger = get_logger(output_dir / f"fold{args.fold}_output.log")
def log(msg): _logger.info(msg)
def log_w(msg): _logger.warn(msg)
log(f'EXP {EXP_ID} start')

def main():
    config = utils.load_config(args.config)
    # copy args to config
    config["mode"] = args.mode
    config["fold"] = args.fold
    config["apex"] = args.apex
    config["output"] = args.output
    config["snapshot"] = args.snapshot

    utils.set_seed(SEED)
    device = torch.device(DEVICE)

    log(f"Fold {args.fold}")

    model = get_img_model(config).to(device)

    log(f"Model type: {model.__class__.__name__}")
    if config["mode"] == 'train':
        train(config, model)
    elif config["mode"] == 'valid':
        valid(config, model)


def valid(cfg, model):
    assert cfg["output"]
    criterion = ImgLoss()
    utils.load_model(cfg["snapshot"], model)
    dataset_valid = RsnaDataset(cfg["fold"], "valid")
    loader_valid = DataLoader(dataset_valid, batch_size=224, shuffle=False, pin_memory=True, num_workers=4)
    with torch.no_grad():
        results = run_nn(cfg, 'valid', model, loader_valid, criterion=criterion)
    utils.save_pickle(results, cfg["output"])
    log('saved to %s' % cfg["output"])


def train(cfg, model):
    criterion = ImgLoss()
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, milestones=[5], gamma=0.5)
    best = {
        'loss': float('inf'),
        'score': 0.0,
        'epoch': -1,
    }
    if "resume_from" in cfg.keys():
        detail = utils.load_model(cfg["resume_from"], model, optim=optim)
        best.update({
            'loss': detail['loss'],
            'score': detail['score'],
            'epoch': detail['epoch'],
        })
    # for param_group in optim.param_groups:
    #     param_group['lr'] = 1e-3 * 0.5
    # log(f"initial lr {utils.get_lr(optim)}")

    dataset_train = RsnaDataset(cfg["fold"], "train")
    dataset_valid = RsnaDataset(cfg["fold"], "valid")
    loader_train = DataLoader(dataset_train, batch_size=56, shuffle=True, pin_memory=True)
    loader_valid = DataLoader(dataset_valid, batch_size=96, shuffle=False, pin_memory=True)

    log('train data: loaded %d records' % len(loader_train.dataset))
    log('valid data: loaded %d records' % len(loader_valid.dataset))

    log('apex %s' % cfg["apex"])
    if cfg["apex"]:
        amp.initialize(model, optim, opt_level='O1')

    for epoch in range(best['epoch']+1, cfg["epoch"]):

        log(f'\n----- epoch {epoch} -----')

        run_nn(cfg, 'train', model, loader_train, criterion=criterion, optim=optim, apex=cfg["apex"])

        with torch.no_grad():
            val = run_nn(cfg, 'valid', model, loader_valid, criterion=criterion)

        detail = {
            'score': val['score'],
            'loss': val['loss'],
            'epoch': epoch,
        }
        if val['loss'] <= best['loss']:
            best.update(detail)

        utils.save_model(model, optim, detail, cfg["fold"], output_dir)

        log('[best] ep:%d loss:%.4f score:%.4f' % (best['epoch'], best['loss'], best['score']))
            
        #scheduler.step(val['loss']) # reducelronplateau
        scheduler.step()

def run_nn(cfg, mode, model, loader, criterion=None, optim=None, scheduler=None, apex=None):
    if mode in ['train']:
        model.train()
    elif mode in ['valid', 'test']:
        model.eval()
    else:
        raise RuntimeError('Unexpected mode %s' % mode)

    t1 = time.time()
    losses = []
    ids_all = []
    targets_all = []
    outputs_all = []

    for i, (inputs, targets, ids) in enumerate(loader):

        batch_size = len(inputs)

        inputs = inputs.cuda()
        for k in targets.keys():
            targets[k] = targets[k].cuda()
        outputs = model(inputs)

        if mode in ['train', 'valid']:
            loss = criterion(outputs, targets)
            with torch.no_grad():
                losses.append(loss.item())

        if mode in ['train']:
            if apex:
                with amp.scale_loss(loss, optim) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward() # accumulate loss
            if (i+1) % cfg["n_grad_acc"] == 0:
                optim.step() # update
                optim.zero_grad() # flush
            
        with torch.no_grad():
            ids_all.extend(ids)
            if mode != 'test':
                targets_all.extend(targets["pe_present_on_image"].cpu().numpy())
            outputs_all.extend(torch.sigmoid(outputs["pe_present_on_image"]).cpu().numpy())
            #outputs_all.append(torch.softmax(outputs, dim=1).cpu().numpy())

        elapsed = int(time.time() - t1)
        eta = int(elapsed / (i+1) * (len(loader)-(i+1)))
        progress = f'\r[{mode}] {i+1}/{len(loader)} {elapsed}(s) eta:{eta}(s) loss:{(np.sum(losses)/(i+1)):.6f} loss200:{(np.sum(losses[-200:])/(min(i+1,200))):.6f} lr:{utils.get_lr(optim):.2e} '
        print(progress, end='')
        sys.stdout.flush()

    result = {
        'ids': ids_all,
        'targets': np.array(targets_all),
        'outputs': np.array(outputs_all),
        'loss': np.sum(losses) / (i+1),
    }

    if mode in ['train', 'valid']:
        result.update(calc_acc(result['targets'], result['outputs']))
        result['score'] = result['acc']

        # log(progress + ' auc:%.4f micro:%.4f macro:%.4f' % (result['auc'], result['auc_micro'], result['auc_macro']))
        log(progress + ' acc:%.4f' % (result['acc']))
        log('ave_loss:%.6f' % (result['loss']))
    else:
        log('')

    return result

# metric functions. return {"metric_name": val}
def calc_acc(targets, outputs):
    cor = np.sum(targets == np.round(outputs))
    return {"acc": cor / float(len(targets))}


if __name__ == "__main__":
    main()
