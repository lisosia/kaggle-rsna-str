#!/usr/bin/env python3

import argparse
from collections import defaultdict
from pathlib import Path
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_recall_fscore_support, log_loss
import torch
import torch.optim
from torch.utils.data import DataLoader

import src.configuration as C
from src.models import get_img_model
import src.utils as utils
from src.utils import get_logger
from src.criterion import ImgLoss
from src.datasets import RsnaDataset, RsnaDataset3D
import src.factory as factory


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=['train', 'valid'], help="train valid")
    parser.add_argument("config", help="Config file path")
    parser.add_argument("--fold", type=int, default=0, help="fold")
    parser.add_argument("--apex", action='store_true', default=False, help="apex")
    parser.add_argument("--output", "-o", help="output path for validation")
    parser.add_argument("--snapshot", "-s", help="snapshot weight path")
    parser.add_argument("--resume-from", help="snapshot to resume train")
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
    config["resume_from"] = args.resume_from

    utils.set_seed(SEED)
    device = torch.device(DEVICE)

    log(f"Fold {args.fold}")

    model = factory.get_model(config).to(device)

    log(f"Model type: {model.__class__.__name__}")
    if config["mode"] == 'train':
        train(config, model)
    elif config["mode"] == 'valid':
        valid(config, model)


def valid(cfg, model):
    assert cfg["output"]
    criterion = factory.get_criterion(cfg)
    utils.load_model(cfg["snapshot"], model)
    loader_valid = factory.get_loader_valid(cfg)
    with torch.no_grad():
        results = run_nn(cfg, 'valid', model, loader_valid, criterion=criterion)
    utils.save_pickle(results, cfg["output"])
    log('saved to %s' % cfg["output"])


def train(cfg, model):
    criterion = factory.get_criterion(cfg)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)

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

    # # to set lr manually after resumed
    # for param_group in optim.param_groups:
    #     param_group['lr'] = 1e-3 * 0.5
    # log(f"initial lr {utils.get_lr(optim)}")

    scheduler, is_reduce_lr = factory.get_scheduler(cfg, optim)
    log(f"is_reduce_lr: {is_reduce_lr}")

    loader_train = factory.get_loader_train(cfg)
    loader_valid = factory.get_loader_valid(cfg)

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

        if is_reduce_lr:
            scheduler.step(val['loss']) # reducelronplateau
        else:
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
    targets_all = defaultdict(list)
    outputs_all = defaultdict(list)

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
            for _k in outputs.keys():  # iter over output keys
                if mode != 'test':
                    targets_all[_k].extend(targets[_k].cpu().numpy())
                outputs_all[_k].extend(torch.sigmoid(outputs[_k]).cpu().numpy())
            #outputs_all.extend(torch.sigmoid(outputs["pe_present_on_image"]).cpu().numpy())
            #outputs_all.append(torch.softmax(outputs, dim=1).cpu().numpy())

        elapsed = int(time.time() - t1)
        eta = int(elapsed / (i+1) * (len(loader)-(i+1)))
        progress = f'\r[{mode}] {i+1}/{len(loader)} {elapsed}(s) eta:{eta}(s) loss:{(np.sum(losses)/(i+1)):.6f} loss200:{(np.sum(losses[-200:])/(min(i+1,200))):.6f} lr:{utils.get_lr(optim):.2e} '
        print(progress, end='')
        sys.stdout.flush()

    result = {
        'ids': ids_all,
        'targets': dict([(k, np.array(v)) for k, v in targets_all.items()]),
        'outputs': dict([(k, np.array(v)) for k, v in outputs_all.items()]),
        'loss': np.sum(losses) / (i+1),
    }

    if mode in ['train', 'valid']:
        SCORE_KEY = "logloss_indeterminate"
        result.update(calc_acc_ind(result['targets'], result['outputs']))
        result.update(calc_f1_ind(result['targets'], result['outputs']))
        result.update(calc_logloss_ind(result['targets'], result['outputs']))

        result['score'] = result[SCORE_KEY]

        # SHOW_KEYS = ["acc_indeterminate", "acc_qa_contrast", "acc_qa_motion"]
        SHOW_KEYS = [k for k in result.keys() if not k in ['ids', 'targets', 'outputs', 'loss']]
        log(progress + ' '.join([k+':%.4f ' % result[k] for k in SHOW_KEYS]))
        log('ave_loss:%.6f' % (result['loss']))
    else:
        log('')

    return result

# metric functions. return {"metric_name": val}
def calc_acc(targets, outputs):
    cor = np.sum(targets == np.round(outputs))
    return {"acc": cor / float(len(targets))}

def calc_acc_ind(targets, outputs):
    ret = {}
    for k in ["indeterminate", "qa_contrast", "qa_motion"]:
        cor = np.sum(targets[k] == np.round(outputs[k]))
        ret["acc_" + k] = cor / float(len(targets[k]))
    return ret
def calc_f1_ind(targets, outputs):
    ret = {}
    for k in ["indeterminate", "qa_contrast", "qa_motion"]:
        pre, rec, f1, _ = precision_recall_fscore_support(targets[k], np.round(outputs[k]), average='binary')
        ret["pre_" + k] = pre
        ret["rec_" + k] = rec
        ret["f1_" + k] = f1
    return ret
def calc_logloss_ind(targets, outputs):
    ret = {}
    for k in ["indeterminate", "qa_contrast", "qa_motion"]:
        ret["logloss_" + k] = log_loss(targets[k], outputs[k], eps=1e-2)
    return ret


if __name__ == "__main__":
    main()
