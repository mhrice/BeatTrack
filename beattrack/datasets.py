import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torchaudio
from torchaudio.transforms import MelSpectrogram, Resample
import torch.nn.functional as F
from pathlib import Path
import pytorch_lightning as pl
from typing import Any, List
from einops import rearrange
from tqdm import tqdm
from cfg import spec, sample_rate, data_length


class BallroomDataset(Dataset):
    def __init__(self, root: str, render: bool = True):
        super().__init__()
        self.root = Path(root)
        self.audio_root = self.root / "BallroomData"
        self.label_root = self.root / "BallroomAnnotations"
        self.audio_files = list(self.root.glob("**/*.wav"))

        self.input_root = self.root / "inputs"
        self.input_root.mkdir(exist_ok=True)
        print("Found {} audio files".format(len(self.audio_files)))
        # Ballfroom dataset is 44100Hz
        self.resample = Resample(orig_freq=44100, new_freq=sample_rate)
        self.mel_spec = MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=spec["n_fft"],
            hop_length=spec["hop_size"],
            win_length=spec["win_length"],
            n_mels=spec["n_mels"],
        )

        if not render:
            return
        for i, f in enumerate(tqdm(self.audio_files)):
            audio, sr = torchaudio.load(f)
            audio = self.resample(audio)
            # Pad or trim to 30 seconds
            if audio.shape[1] < data_length:
                audio = F.pad(audio, (0, data_length - audio.shape[1]))
            else:
                audio = audio[:, :data_length]
            mel = self.mel_spec(audio)
            channels, bins, frames = mel.shape
            mel = rearrange(mel, "c b f -> f (b c)")

            label_file = self.label_root / f"{f.stem}.beats"
            beats, downbeats = label2vec(label_file, spec["hop_size"], frames)
            torch.save(mel, self.input_root / f"{i}_mel.pt")
            torch.save(beats, self.input_root / f"{i}_beats.pt")
            torch.save(downbeats, self.input_root / f"{i}_downbeats.pt")
            torch.save(f, self.input_root / f"{i}_path.pt")

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx: int):
        mel = torch.load(self.input_root / f"{idx}_mel.pt")
        beats = torch.load(self.input_root / f"{idx}_beats.pt")
        downbeats = torch.load(self.input_root / f"{idx}_downbeats.pt")
        return mel.unsqueeze(0), beats, downbeats


def label2vec(
    label_file: Path, hop_size: int, num_frames: int, widen_length: int = 2
) -> torch.Tensor:
    beats = [0] * num_frames
    downbeats = [0] * num_frames
    for line in label_file.open("r"):
        # Fix for file that has tab instead of space
        time, beat_num = line.replace("\t", " ").split(" ")
        frame_id = round(float(time) * sample_rate / spec["hop_size"])
        if frame_id < num_frames:
            beats[frame_id] = 1
            if beat_num.strip() == "1":
                downbeats[frame_id] = 1
            # Add 0.5 to the left and right of the label
            for i in range(1, widen_length + 1):
                if frame_id - i > 0:
                    beats[frame_id - i] = 0.5
                    if beat_num.strip() == "1":
                        downbeats[frame_id - i] = 0.5
                if frame_id + i < len(beats) - 1:
                    beats[frame_id + i] = 0.5
                    if beat_num.strip() == "1":
                        downbeats[frame_id + i] = 0.5

    return torch.Tensor(beats), torch.Tensor(downbeats)


class BallroomDatamodule(pl.LightningDataModule):
    def __init__(
        self,
        dataset,
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
