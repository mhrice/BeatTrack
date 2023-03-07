sample_rate = 44100
data_length = 30 * sample_rate  # 30 seconds of audio

cpu_cfg = {
    "batch_size": 1,
    "num_workers": 0,
    "wandb": True,
    "max_epochs": 50,
    "gpu": False,
}

gpu_cfg = {
    "batch_size": 32,
    "num_workers": 0,
    "wandb": True,
    "max_epochs": 1000,
    "gpu": True,
}

spec = {
    "n_fft": 2048,
    "win_length": None,
    "hop_size": round(sample_rate / 100),  # 10ms hop size
    "n_mels": 81,
}
