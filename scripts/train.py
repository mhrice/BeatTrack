from beattrack.model import BeatTCN
from beattrack.datasets import BallroomDataset, BallroomDatamodule
from beattrack.callbacks import callbacks
import pytorch_lightning as pl
from pytorch_lightning.utilities.model_summary import ModelSummary
from pytorch_lightning.loggers import WandbLogger
import shutil
import os
import yaml
import torch


def main():
    if torch.cuda.is_available():
        cfg = yaml.load(open("config.yaml", "r"), Loader=yaml.FullLoader)
    pl.seed_everything(123)
    dataset = BallroomDataset(root="data", render=True)
    datamodule = BallroomDatamodule(
        dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=True,
        persistent_workers=True,
    )
    logger = None
    if cfg.wandb:
        logger = WandbLogger(project="beattrack", entity="mattricesound", save_dir=".")
    if os.path.exists("./ckpts") and os.path.isdir("./ckpts"):
        shutil.rmtree("./ckpts")

    model = BeatTCN()
    trainer = pl.Trainer(
        max_epochs=cfg.max_epochs,
        logger=logger,
        log_every_n_steps=1,
        callbacks=callbacks,
    )
    trainer.fit(model, datamodule)
    trainer.test(model, datamodule, ckpt_path="best")


if __name__ == "__main__":
    main()
