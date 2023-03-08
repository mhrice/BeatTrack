from beattrack.model import BeatTCN
from beattrack.datasets import BallroomDataset, BallroomDatamodule
from beattrack.callbacks import callbacks
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import shutil
import os
import torch
from cfg import cpu_cfg, gpu_cfg
import argparse


def main():
    parser = argparse.ArgumentParser(description="Train BeatTCN")
    parser.add_argument("--skip_render", action="store_true", default=False)
    args = parser.parse_args()

    # Use GPU config if available
    if torch.cuda.is_available():
        cfg = gpu_cfg
        torch.set_float32_matmul_precision("medium")
    else:
        cfg = cpu_cfg
    pl.seed_everything(123)
    # Load Datamodule
    dataset = BallroomDataset(root="data", render=not args.skip_render)
    datamodule = BallroomDatamodule(
        dataset,
        batch_size=cfg["batch_size"],
        num_workers=cfg["num_workers"],
        pin_memory=True,
        persistent_workers=True,
    )
    # Create Logger
    logger = None
    if cfg["wandb"]:
        logger = WandbLogger(project="beattrack", entity="mattricesound", save_dir=".")
    if os.path.exists("./ckpts") and os.path.isdir("./ckpts"):
        shutil.rmtree("./ckpts")

    model = BeatTCN()
    if cfg["gpu"]:
        accelerator = "gpu"
    else:
        accelerator = None
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=cfg["max_epochs"],
        logger=logger,
        log_every_n_steps=1,
        callbacks=callbacks,
        accelerator=accelerator,
        devices=1,
    )
    # Train
    trainer.fit(model, datamodule)
    trainer.test(model, datamodule, ckpt_path="best")


if __name__ == "__main__":
    main()
