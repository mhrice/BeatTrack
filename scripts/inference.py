import pytorch_lightning as pl
from beattrack.model import BeatNetWrapper
import torchaudio

model = BeatNetWrapper.load_from_checkpoint(
    "ckpts/version_18/checkpoints/epoch=9-step=2999.ckpt"
)
model.eval()

x, sr = torchaudio.load("data/ballroom/1/1.wav")

y_hat = model(x)
