# Spotify Audio Analysis
Spotify Audio Analysis Repo for CS 5440

# Meeting Room
Team meeting room is **202-A**.

# (Violet) Team Members
 - Imbleau, Spencer
 - Frazier, Maclean
 - Lapenseee-Rankine, Danielle
 - Reeves, Andrew
 - Lovingood, Jake

# Tools, IDEs, Environments
 - Python 3.7+
 - Anaconda 3 - [Install](https://docs.anaconda.com/anaconda/install/)
 - PyCharm Community Edition - [Install](https://www.jetbrains.com/pycharm/download/)

# Getting started
 - After cloning, you **must** put `spotify_train.npz` and `spotify_valid.npz` in the `data/` folder! These are too big for GitHub to track.
 - If the conda environment has never been created: ` conda env create -f cs5440.yml`
 - Activating the conda environment: `conda activate cs5440`
 - Program run example: `python3 loudness.py`

# Troubleshooting
 - `ImportError: cannot import name seaborn`
   - Fix: `pip install seaborn`
 - `"AttributeError: 'str' object has no attribute 'decode' " , while Loading a Keras Saved Model`
   - Fix: `pip install 'h5py==2.10.0' --force-reinstall`
 
# WebCAT
[http://webcatvm.cs.appstate.edu:8080/Web-CAT](http://webcatvm.cs.appstate.edu:8080/Web-CAT)
