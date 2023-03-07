# TCN-based Joint Beat and Downbeat Tracking
A pytorch-lightning implementation of TCN-based Joint Beat and Downbeat Tracking model described in the paper [Temporal convolutional networks for musical audio beat tracking](https://ieeexplore.ieee.org/document/8902578), with some extra ideas from [Deconstruct, Analyse, Reconstruct: How to improve Tempo, Beat, and Downbeat Estimation](https://program.ismir2020.net/static/final_papers/223.pdf).

# Setup
## Install Dependencies
1. `git clone https://github.com/mhrice/BeatTrack.git`
2. `cd BeatTrack`
1. `python3 -m venv env`
2. `source env/bin/activate`
3. `pip install cython numpy`
4. `pip install -e .`

Need a 2 stage pip install because of madmom issues

## Fix Other Madmom issue
In newer versions of python, madmom has an issue with the `processors.py` file. To fix this, run the following command, replacing `{python-version}` with your python version:
`cp processors.py env/lib/{python-version}/site-packages/madmom/processors.py`

## Download Ballroom Dataset (not needed for inference)
1. `mkdir data && cd data`
2. `wget http://mtg.upf.edu/ismir2004/contest/tempoContest/data1.tar.gz`
2. `git clone https://github.com/CPJKU/BallroomAnnotations.git`
3. `tar -xvzf data1.tar.gz`
4. `rm data1.tar.gz`


# Training
`python scripts/train.py`

# Inference
`python scripts/inference.py audio_file_path {checkpoint_path}`

or
```
from scripts.inference import beatTracker
beatTracker(audio_file_path)
```

# Parameters
See `cfg.py` for all parameters.

# Checkpoints
`checkpoints/best.ckpt` is a checkpoint where the model was trained to do joint beat/downbeat predictions on the Ballroom dataset for 165 epochs.

