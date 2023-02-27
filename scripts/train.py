from beattrack.model import BeatNet
from beattrack.datasets import BallroomDataset, BallroomDatamodule
import pytorch_lightning as pl
from pytorch_lightning.utilities.model_summary import ModelSummary
from pytorch_lightning.loggers import WandbLogger


def main():
    pl.seed_everything(123)
    dataset = BallroomDataset(root="data/ballroom", render=False)
    datamodule = BallroomDatamodule(
        dataset,
        root="data/ballroom",
        batch_size=32,
        num_workers=0,
        pin_memory=True,
        persistent_workers=True,
    )

    logger = WandbLogger(project="beattrack", entity="mattricesound", save_dir=".")
    callbacks = []
    model_checkpoint = pl.callbacks.ModelCheckpoint(
        monitor="valid_loss",
        save_top_k=1,
        save_last=True,
        mode="min",
        verbose=False,
        dirpath="./ckpts2/",
        filename="{epoch:02d}-{valid_loss:.3f}",
    )
    callbacks.append(model_checkpoint)
    model = BeatNet()
    trainer = pl.Trainer(
        max_epochs=10, logger=logger, log_every_n_steps=1, callbacks=callbacks
    )
    summary = ModelSummary(model)
    print(summary)
    trainer.fit(model, datamodule)


if __name__ == "__main__":
    main()
