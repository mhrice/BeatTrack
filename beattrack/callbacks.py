import pytorch_lightning as pl


model_checkpoint = pl.callbacks.ModelCheckpoint(
    monitor="valid_loss",
    save_top_k=1,
    save_last=True,
    mode="min",
    verbose=False,
    dirpath="./ckpts",
    filename="{epoch:02d}-{valid_loss:.3f}",
)
learning_rate_monitor = pl.callbacks.LearningRateMonitor(logging_interval="step")
early_stop_callback = pl.callbacks.EarlyStopping(
    monitor="valid_loss",
    min_delta=0.01,
    patience=100,
    mode="min",
    verbose=False,
)

callbacks = [model_checkpoint, learning_rate_monitor, early_stop_callback]
