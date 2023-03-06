## Install Packages
1. `python3 -m venv env`
2. `source env/bin/activate`
3. `pip install cython`
4. `pip install -e .`

## Download Data
1. `mkdir data && cd data`
2. `wget http://mtg.upf.edu/ismir2004/contest/tempoContest/data1.tar.gz`
2. `git clone https://github.com/CPJKU/BallroomAnnotations.git`
3. `tar -xvzf data1.tar.gz`
4. `rm data1.tar.gz`

## Fix Madmom issue
`cp processors.py env/lib/python3.10/site-packages/madmom/processors.py`