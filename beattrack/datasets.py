import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torchaudio
from torchaudio.transforms import MelSpectrogram
import torch.nn.functional as F
from pathlib import Path
import pytorch_lightning as pl
from typing import Any, List
from einops import rearrange
from tqdm import tqdm

n_fft = 2048
win_length = None
sample_rate = 44100
hop_size = round(sample_rate / 100)  # 10ms hop size
n_mels = 81
window = torch.hann_window(n_fft)
data_length = 30 * sample_rate  # 30 seconds of audio


class BallroomDataset(Dataset):
    def __init__(self, root: str, render: bool = True):
        super().__init__()
        self.root = Path(root)
        self.audio_files = list(self.root.glob("**/*.wav"))
        self.label_root = self.root / "beats"
        self.input_root = self.root / "inputs"
        self.input_root.mkdir(exist_ok=True)
        print("Found {} audio files".format(len(self.audio_files)))
        self.mel_spec = MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_size,
            win_length=win_length,
            n_mels=n_mels,
        )

        if not render:
            return
        total = 0
        for i, f in enumerate(tqdm(self.audio_files)):
            audio, sr = torchaudio.load(f)
            # Pad or trim to 30 seconds
            if audio.shape[1] < data_length:
                audio = F.pad(audio, (0, data_length - audio.shape[1]))
            else:
                audio = audio[:, :data_length]
            mel = self.mel_spec(audio)
            channels, bins, frames = mel.shape
            mel = rearrange(mel, "c b f -> f (b c)")

            label_file = self.label_root / f"{f.stem}.beats"
            label = label2vec(label_file, hop_size, frames)
            torch.save(mel, self.input_root / f"{i}_mel.pt")
            torch.save(label, self.input_root / f"{i}_lab.pt")
            self.total = total

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx: int):
        mel = torch.load(self.input_root / f"{idx}_mel.pt")
        label = torch.load(self.input_root / f"{idx}_lab.pt")
        return mel.unsqueeze(0), label


def label2vec(label_file: Path, hop_size: int, num_frames: int) -> torch.Tensor:
    labels = [0] * num_frames
    for line in label_file.open("r"):
        time, beat_num = line.split(" ")
        frame_id = round(float(time) * sample_rate / hop_size)
        if frame_id < num_frames:
            labels[frame_id] = 1
    return torch.tensor(labels)


class BallroomDatamodule(pl.LightningDataModule):
    def __init__(
        self,
        dataset,
        root: str,
        batch_size: int,
        num_workers: int,
        pin_memory: bool = False,
        **kwargs: int,
    ) -> None:
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.root = Path(root)

    def setup(self, stage: Any = None) -> None:
        train_split = 0.8
        val_split = 0.1
        test_split = 0.1

        self.data_train, self.data_val, self.data_test = random_split(
            self.dataset, [train_split, val_split, test_split]
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )
