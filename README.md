# Calcium Analysis

This is the official repository for the paper **Quantitative Analysis of Miniature Synaptic Calcium Transients Using Positive Unlabeled Deep Learning**  

Open in [Colab]()


# System Requirements

## Hardware Requirements
Training and inference of the deep learning models require a GPU for reasonable run time. 

## Software Requirements
### OS requirements 
The source code was tested on Linux - CentOS 7
### Python dependencies 
The source code was tested on Python 3.8.10. All required libraries are listed in the `requirements.txt` file.

# Installation
Clone the repo and move into the root directory

```bash
git clone https://github.com/FLClab/Calcium-Analysis.git
cd CalciumAnalysis/experiments
```

Create a virtual environment and install the required packages 
```bash
python --version # make sure you have Python 3.8
python -m venv calcium-env # create a virtual environment named 'calcium-env`
source ./calcium-env/bin/activate # activate the environment
pip install -r requirements.txt # install required packages
```

# Documentation
## 3D U-Net and StarDist-3D for mSCT segmentation
### Datasets
### Models and baselines

## Training and testing on your own calcium videos

## Reproducing the published results

# Citation
