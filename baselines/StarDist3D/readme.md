# StarDist3D Baseline

This folder contains the scripts used to train StarDist3D.

## Installation

Follow the installation process given in the [StarDist3D repository](https://github.com/stardist/stardist).

## Training

The training can be launched using the provided `train.py` file

```bash
python train.py --config ./configs/subset_0.25_1-0/config-stardist3d_subset_0.yml
```

## Data

It is assumed that there exist a folder `data`  which contains the dataset `80-20_calcium-dataset.h5`.
