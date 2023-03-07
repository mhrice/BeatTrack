from beattrack.model import BeatTCN
import torchaudio
from torchaudio.transforms import MelSpectrogram
from pathlib import Path
from einops import rearrange
import torch
import torch.nn.functional as F
from madmom.features import DBNBeatTrackingProcessor
from cfg import spec, sample_rate, data_length
import sys

# Global Tools
mel_spec = MelSpectrogram(
    sample_rate=sample_rate,
    n_fft=spec["n_fft"],
    hop_length=spec["hop_size"],
    win_length=spec["win_length"],
    n_mels=spec["n_mels"],
)
beat_dbn = DBNBeatTrackingProcessor(
    min_bpm=55, max_bpm=215, transition_labmda=100, fps=100
)
downbeat_dbn = DBNBeatTrackingProcessor(
    min_bpm=15, max_bpm=80, transition_labmda=100, fps=100
)
model = BeatTCN.load_from_checkpoint("checkpoints/best.ckpt")
model.eval()


def beatTracker(inputFile: str, checkpoint: str = None):
    global model
    file_path = Path(inputFile)
    mel = preprocess(file_path)
    if checkpoint is not None:
        model = BeatTCN.load_from_checkpoint(checkpoint)
        model.eval()
    with torch.no_grad():
        preds = model(mel)
    # Split into beat and downbeat predictions
    beat_preds, downbeat_preds = preds.split(1, dim=1)
    # Get beat and downbeat times using DBN
    beat_times = beat_dbn(beat_preds.cpu().view(-1))
    downbeat_times = downbeat_dbn(downbeat_preds.cpu().view(-1))
    return beat_times, downbeat_times


def preprocess(audio_path):
    # Load audio and resample
    audio, sr = torchaudio.load(audio_path)
    resample = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sample_rate)
    audio = resample(audio)
    # Pad or trim to 30 seconds
    if audio.shape[1] < data_length:
        audio = F.pad(audio, (0, data_length - audio.shape[1]))
    else:
        audio = audio[:, :data_length]
    # Get mel spectrogram
    mel = mel_spec(audio)
    channels, bins, frames = mel.shape
    # Reshape to (batch, channels, frames, bins)
    mel = rearrange(mel, "c b f -> f (b c)")
    mel = mel.view(1, 1, mel.shape[0], mel.shape[1])
    return mel


if __name__ == "__main__":
    audio_path = sys.argv[1]
    beat_times, downbeat_times = beatTracker(audio_path)
    print(f"{beat_times=}")
    print(f"{downbeat_times=}")
