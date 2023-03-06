from audiomentations import (
    Compose,
    SpecCompose,
    AddGaussianNoise,
    PitchShift,
    HighPassFilter,
    LowPassFilter,
    TanhDistortion,
    PolarityInversion,
    SpecChannelShuffle,
    SpecFrequencyMask,
)

waveform_augment = Compose(
    [
        HighPassFilter(min_cutoff_freq=80, max_cutoff_freq=200, p=0.25),
        LowPassFilter(min_cutoff_freq=8000, max_cutoff_freq=16000, p=0.25),
        PitchShift(min_semitones=-8, max_semitones=8, p=0.5),
        AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.05),
        TanhDistortion(min_distortion=0.1, max_distortion=0.7, p=0.2),
        PolarityInversion(p=0.5),
    ]
)

spectral_augment = SpecCompose(
    [
        SpecChannelShuffle(p=0.5),
        SpecFrequencyMask(p=0.5),
    ]
)
