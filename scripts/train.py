from beattrack.model import BeatNet
from beattrack.datasets import BallroomDataset, BallroomDatamodule
from beattrack.callbacks import callbacks
import pytorch_lightning as pl
from pytorch_lightning.utilities.model_summary import ModelSummary
from pytorch_lightning.loggers import WandbLogger
import shutil
import os


def main():
    pl.seed_everything(123)
    dataset = BallroomDataset(root="data/ballroom", render=True)
    datamodule = BallroomDatamodule(
        dataset,
        batch_size=1,
        num_workers=0,
        pin_memory=True,
        persistent_workers=True,
    )

    logger = WandbLogger(project="beattrack", entity="mattricesound", save_dir=".")
    if os.path.exists("./ckpts") and os.path.isdir("./ckpts"):
        shutil.rmtree("./ckpts")

    model = BeatNet()
    trainer = pl.Trainer(
        max_epochs=50, logger=logger, log_every_n_steps=1, callbacks=callbacks
    )
    summary = ModelSummary(model)
    print(summary)
    trainer.fit(model, datamodule)
    trainer.test(model, datamodule, ckpt_path="best")


if __name__ == "__main__":
    main()
