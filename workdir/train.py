import src.callbacks as clb
import src.configuration as C
from src.models import get_img_model
import src.utils as utils

from catalyst.dl import SupervisedRunner

from pathlib import Path
import pandas as pd

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", required=True, help="Config file path")
    parser.add_argument("--fold", type=int, nargs="+", default=0, help="Config file path")
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    config = utils.load_config(args.config)

    global_params = config["globals"]

    output_dir = Path(global_params["output_dir"])
    output_dir.mkdir(exist_ok=True, parents=True)
    logger = utils.get_logger(output_dir / "output.log")

    utils.set_seed(global_params["seed"])
    device = torch.device(global_params["device"])

    df, datadir = C.get_metadata(config)

    for i in range(5):
        if i not in global_params["folds"]:
            continue
        logger.info("=" * 20)
        logger.info(f"Fold {i}")
        logger.info("=" * 20)
        if "task" not in global_params.keys() or global_params["task"] != "nocall":
            # trn_df = df.loc[trn_idx, :].reset_index(drop=True)
            # val_df = df.loc[val_idx, :].reset_index(drop=True)
            trn_df = df[df.fold!=i].reset_index(drop=True)
            val_df = df[df.fold==i].reset_index(drop=True)

            loaders = {
                phase: C.get_loader(df_, datadir, config, phase)
                for df_, phase in zip([trn_df, val_df], ["train", "valid"])
            }
        else:
            loaders = C.get_loaders_nocall(config)

        model = get_img_model(config).to(device)
        criterion = C.get_criterion(config).to(device)
        optimizer = C.get_optimizer(model, config)
        scheduler = C.get_scheduler(optimizer, config)
        callbacks = clb.get_callbacks(config)

        if True:  # resume
            import torch
            logger.warn("Load from checkpoint")
            # checkpoint = torch.load("output/007_ResNet18_Simple_InitialSeg/fold0/checkpoints/train.19_full.pth")
            checkpoint = torch.load("output/057_linpool_20s_aug2_normfix/fold0/bak/train.9_full.pth")
            model.load_state_dict(checkpoint["model_state_dict"])

        logger.info(f"Model type: {model.__class__.__name__}")

        runner = SupervisedRunner(
            device=device,
            #input_key=global_params["input_key"],
            #input_target_key=global_params["input_target_key"])
        runner.train(
            fp16=None,
            model=model,
            criterion=criterion,
            loaders=loaders,
            optimizer=optimizer,
            scheduler=scheduler,
            num_epochs=global_params["num_epochs"],
            verbose=True,
            logdir=output_dir / f"fold{i}",
            callbacks=callbacks,
            main_metric=global_params["main_metric"],
            minimize_metric=global_params["minimize_metric"])
