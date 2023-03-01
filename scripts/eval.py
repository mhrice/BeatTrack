from beattrack.model import BeatNet
import torchaudio
from torchaudio.transforms import MelSpectrogram
from pathlib import Path
from einops import rearrange
import torch
import torch.nn.functional as F
from madmom.features import DBNBeatTrackingProcessor
import mir_eval
import numpy as np
from tqdm import tqdm

n_fft = 2048
win_length = None
sample_rate = 44100
hop_size = round(sample_rate / 100)  # 10ms hop size
n_mels = 81
window = torch.hann_window(n_fft)
data_length = 30 * sample_rate  # 30 seconds of audio
mel_spec = MelSpectrogram(
    sample_rate=sample_rate,
    n_fft=n_fft,
    hop_length=hop_size,
    win_length=win_length,
    n_mels=n_mels,
)


def main():
    model = BeatNet.load_from_checkpoint("ckpts/last.ckpt")
    model = model.eval()

    dbn = DBNBeatTrackingProcessor(
        min_bpm=55, max_bpm=215, transition_labmda=100, fps=100
    )

    root = Path("data/ballroom")
    f_mes = []
    cmlc = []
    cmlt = []
    amlc = []
    amlt = []
    d = []

    files = list(root.glob("**/*.wav"))
    for f in tqdm(root.glob("**/*.wav"), total=len(files)):
        mel = preprocess(f)
        mel = mel.unsqueeze(0).unsqueeze(0)

        with torch.no_grad():
            beats = model(mel)
        beat_times = dbn(beats.view(-1))

        label_root = root / "beats"
        label_file = label_root / f"{f.stem}.beats"
        label = label2vec(label_file, hop_size, mel.shape[2])

        gt_times = []
        for i, lab in enumerate(label):
            if lab == 1:
                gt_times.append(i * hop_size / sample_rate)
        gt_times = np.array(gt_times)
        # loss = F.binary_cross_entropy(beats.squeeze(1), label.view(1, -1).float())
        eval = mir_eval.beat.evaluate(gt_times, beat_times)
        f_mes.append(eval["F-measure"])
        cmlc.append(eval["Correct Metric Level Continuous"])
        cmlt.append(eval["Correct Metric Level Total"])
        amlc.append(eval["Any Metric Level Continuous"])
        amlt.append(eval["Any Metric Level Total"])
        d.append(eval["Information gain"])

    print(f"F-measure: {np.mean(f_mes)}")
    print(f"CMLC: {np.mean(cmlc)}")
    print(f"CMLT: {np.mean(cmlt)}")
    print(f"AMLC: {np.mean(amlc)}")
    print(f"AMLT: {np.mean(amlt)}")
    print(f"Information gain: {np.mean(d)}")


def preprocess(audio_path):
    audio, sr = torchaudio.load(audio_path)
    resample = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sample_rate)
    audio = resample(audio)
    data_length = 30 * sample_rate  # 30 seconds of audio

    # Pad or trim to 30 seconds
    if audio.shape[1] < data_length:
        audio = F.pad(audio, (0, data_length - audio.shape[1]))
    else:
        audio = audio[:, :data_length]
    mel = mel_spec(audio)
    channels, bins, frames = mel.shape
    mel = rearrange(mel, "c b f -> f (b c)")
    return mel


def label2vec(label_file: Path, hop_size: int, num_frames: int) -> torch.Tensor:
    labels = [0] * num_frames
    for line in label_file.open("r"):
        time, beat_num = line.split(" ")
        frame_id = round(float(time) * sample_rate / hop_size)
        if frame_id < num_frames:
            labels[frame_id] = 1
    return torch.tensor(labels)


if __name__ == "__main__":
    main()
