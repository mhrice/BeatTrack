from madmom.features import DBNBeatTrackingProcessor
import mir_eval
import numpy as np
import torch

sample_rate = 44100
hop_size = round(sample_rate / 100)  # 10ms hop size


def eval(
    batch: torch.Tensor,
    beat_preds: torch.Tensor,
    downbeat_preds: torch.Tensor,
    mode: str,
):
    beat_dbn = DBNBeatTrackingProcessor(
        min_bpm=55, max_bpm=215, transition_labmda=100, fps=100
    )
    downbeat_dbn = DBNBeatTrackingProcessor(
        min_bpm=15, max_bpm=80, transition_labmda=100, fps=100
    )
    beat_f_mes = []
    downbeat_f_mes = []
    beat_cmlc = []
    downbeat_cmlc = []
    beat_cmlt = []
    downbeat_cmlt = []
    beat_amlc = []
    downbeat_amlc = []
    beat_amlt = []
    downbeat_amlt = []
    beat_d = []
    downbeat_d = []

    mel, beat_label, downbeat_label = batch

    for beats, downbeats, bl, dl in zip(
        beat_preds, downbeat_preds, beat_label, downbeat_label
    ):
        beat_times = beat_dbn(beats.cpu().view(-1))
        downbeat_times = downbeat_dbn(downbeats.cpu().view(-1))

        beat_gt_times = []
        downbeat_gt_times = []
        for i, lab in enumerate(bl):
            if lab == 1:
                beat_gt_times.append(i * hop_size / sample_rate)
        for i, lab in enumerate(dl):
            if lab == 1:
                downbeat_gt_times.append(i * hop_size / sample_rate)
        beat_gt_times = np.array(beat_gt_times)
        downbeat_gt_times = np.array(downbeat_gt_times)
        beat_evaluation = mir_eval.beat.evaluate(beat_gt_times, beat_times)
        downbeat_evaluation = mir_eval.beat.evaluate(downbeat_gt_times, downbeat_times)
        beat_f_mes.append(beat_evaluation["F-measure"])
        beat_cmlc.append(beat_evaluation["Correct Metric Level Continuous"])
        beat_cmlt.append(beat_evaluation["Correct Metric Level Total"])
        beat_amlc.append(beat_evaluation["Any Metric Level Continuous"])
        beat_amlt.append(beat_evaluation["Any Metric Level Total"])
        beat_d.append(beat_evaluation["Information gain"])

        downbeat_f_mes.append(downbeat_evaluation["F-measure"])
        downbeat_cmlc.append(downbeat_evaluation["Correct Metric Level Continuous"])
        downbeat_cmlt.append(downbeat_evaluation["Correct Metric Level Total"])
        downbeat_amlc.append(downbeat_evaluation["Any Metric Level Continuous"])
        downbeat_amlt.append(downbeat_evaluation["Any Metric Level Total"])
        downbeat_d.append(downbeat_evaluation["Information gain"])

        # Only log first of batch
        if mode == "valid":
            break

    return {
        "Beat F-measure": np.mean(beat_f_mes),
        "Downbeat F-measure": np.mean(downbeat_f_mes),
        "Beat Correct Metric Level Continuous": np.mean(beat_cmlc),
        "Downbeat Correct Metric Level Continuous": np.mean(downbeat_cmlc),
        "Beat Correct Metric Level Total": np.mean(beat_cmlt),
        "Downbeat Correct Metric Level Total": np.mean(downbeat_cmlt),
        "Beat Any Metric Level Continuous": np.mean(beat_amlc),
        "Downbeat Any Metric Level Continuous": np.mean(downbeat_amlc),
        "Beat Any Metric Level Total": np.mean(beat_amlt),
        "Downbeat Any Metric Level Total": np.mean(downbeat_amlt),
        "Beat Information gain": np.mean(beat_d),
        "Downbeat Information gain": np.mean(downbeat_d),
    }
