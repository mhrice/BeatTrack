from madmom.features import DBNBeatTrackingProcessor
import mir_eval
import numpy as np
import torch

sample_rate = 44100
hop_size = round(sample_rate / 100)  # 10ms hop size


def eval(model: torch.nn.Module, test_data: torch.Tensor):
    dbn = DBNBeatTrackingProcessor(
        min_bpm=55, max_bpm=215, transition_labmda=100, fps=100
    )
    f_mes = []
    cmlc = []
    cmlt = []
    amlc = []
    amlt = []
    d = []

    mel, label = test_data
    with torch.no_grad():
        beats = model(mel)
    for beats, label in zip(beats, label):
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
