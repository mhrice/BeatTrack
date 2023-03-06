from madmom.features import DBNBeatTrackingProcessor
import mir_eval
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm

sample_rate = 44100
hop_size = round(sample_rate / 100)  # 10ms hop size


def eval(model: torch.nn.Module, test_dataset: torch.utils.data.Dataset):
    dbn = DBNBeatTrackingProcessor(
        min_bpm=55, max_bpm=215, transition_labmda=100, fps=100
    )
    f_mes = []
    cmlc = []
    cmlt = []
    amlc = []
    amlt = []
    d = []

    for mel, label in tqdm(test_dataset, total=len(test_dataset)):
        mel = mel.unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            beats = model(mel)
        beat_times = dbn(beats.view(-1))

        gt_times = []
        for i, lab in enumerate(label):
            if lab == 1:
                gt_times.append(i * hop_size / sample_rate)
        gt_times = np.array(gt_times)
        eval = mir_eval.beat.evaluate(gt_times, beat_times)
        f_mes.append(eval["F-measure"])
        cmlc.append(eval["Correct Metric Level Continuous"])
        cmlt.append(eval["Correct Metric Level Total"])
        amlc.append(eval["Any Metric Level Continuous"])
        amlt.append(eval["Any Metric Level Total"])
        d.append(eval["Information gain"])
    return {
        "F-measure": np.mean(f_mes),
        "Correct Metric Level Continuous": np.mean(cmlc),
        "Correct Metric Level Total": np.mean(cmlt),
        "Any Metric Level Continuous": np.mean(amlc),
        "Any Metric Level Total": np.mean(amlt),
        "Information gain": np.mean(d),
    }
