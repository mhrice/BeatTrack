from beattrack.model import BeatNetWrapper
from beattrack.datasets import BallroomDataset, BallroomDatamodule
import pytorch_lightning as pl
from pytorch_lightning.utilities.model_summary import ModelSummary
from pytorch_lightning.loggers import WandbLogger


def main():
    pl.seed_everything(123)
    dataset = BallroomDataset(root="data/ballroom", skip_render=True)
    datamodule = BallroomDatamodule(
        dataset,
        root="data/ballroom",
        batch_size=32,
        num_workers=0,
        pin_memory=True,
        persistent_workers=True,
    )

    logger = WandbLogger(project="beattrack", entity="mattricesound", save_dir=".")

    model = BeatNetWrapper()
    trainer = pl.Trainer(
        max_epochs=10, logger=logger, log_every_n_steps=1, default_root_dir="./ckpts"
    )
    summary = ModelSummary(model)
    print(summary)
    trainer.fit(model, datamodule)


if __name__ == "__main__":
    main()
