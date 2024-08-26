<a target="_blank" href="https://colab.research.google.com/github/FLClab/Calcium-Analysis/blob/main/CalciumUNet3D_ZeroCostDL4Mic.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

# Calcium Analysis

This is the official repository for the paper **Quantitative Analysis of Miniature Synaptic Calcium Transients Using Positive Unlabeled Deep Learning**.

Data for this paper, including the raw training input images, the pre-processed training dataset, the pre-trained models, and the testing dataset are available for download [here](https://s3.valeria.science/flclab-calcium/index.html).

## Abstract

Ca<sup>2+</sup> imaging methods are widely used for studying cellular activity in the brain, allowing detailed analysis of dynamic processes across various scales. Enhanced by high-contrast optical microscopy and fluorescent Ca<sup>2+</sup> sensors, this technique can be used to reveal localized Ca<sup>2+</sup> fluctuations within neurons, including in sub-cellular compartments, such as the dendritic shaft or spines. Despite advances in Ca<sup>2+</sup> sensors, the analysis of miniature Synaptic Calcium Transients (mSCTs), characterized by variability in morphology and low signal-to-noise ratios, remains challenging. Traditional threshold-based methods struggle with the detection and segmentation of these small, dynamic events. Deep learning (DL) approaches offer promising solutions but are limited by the need for large annotated datasets. Positive Unlabeled (PU) learning addresses this limitation by leveraging unlabeled instances to increase dataset size and enhance performance. This approach is particularly useful in the case of mSCTs that are scarce and small, associated with a very small proportion of the foreground pixels. PU learning significantly increases the effective size of the training dataset, improving model performance. Here, we present a PU learning-based strategy for detecting and segmenting mSCTs. We evaluate the performance of two 3D deep learning models, StarDist-3D and 3D U-Net, which are well established for the segmentation of small volumetric structures in microscopy datasets. By integrating PU learning, we enhance the 3D U-Net's performance, demonstrating significant gains over traditional methods. This work pioneers the application of PU learning in Ca<sup>2+</sup> imaging analysis, offering a robust framework for mSCT detection and segmentation. We also demonstrate how this quantitative analysis pipeline can be used for subsequent mSCTs feature analysis. We characterize morphological and kinetic changes of mSCTs associated with the application of chemical long-term potentiation (cLTP) stimulation in cultured rat hippocampal neurons. Our data-driven approach shows that a cLTP-inducing stimulus leads to the emergence of new active dendritic regions and differently affects mSCTs subtypes. 

## System Requirements

### Hardware Requirements

Training and inference of the deep learning models require a GPU for reasonable run time.

The models were trained on an high-performance computing system with the following specifications:
- 1 node with 4 CPU cores.
- 1 Tesla V100-SXM2-16GB GPU per node.
- Memory allocation of 64.0 GB.

The time required to train the models depends on the dataset size and the number of steps. Training the 3D U-Net model on the mSCT dataset with 100k steps took approximately 14 hours. Any consumer-grade GPU with at least 8GB of memory should be sufficient for training the models.

### Software Requirements

#### OS requirements 
The source code was tested on Linux - CentOS 7

#### Python dependencies 
The source code was tested on Python 3.8.10 and 3.11.5. The required python installation should be `python>=3.8`. All required libraries are listed in the provided `experiments/requirements.txt` file.

## Installation

### Colab notebook (recommended)

The provided Colab notebook is the easiest way to use the pre-trained models and to train the models on your own calcium videos. All dependencies are already installed in the notebook. The notebook is available [here](https://colab.research.google.com/github/FLClab/Calcium-Analysis/blob/main/CalciumUNet3D_ZeroCostDL4Mic.ipynb).

### From source

Clone the repo and move into the root directory

```bash
git clone https://github.com/FLClab/Calcium-Analysis.git
cd Calcium-Analysis/experiments
```

Create a virtual environment and install the required packages 
```bash
python --version # make sure you have Python 3.11
python -m venv calcium-env # create a virtual environment named 'calcium-env`
source ./calcium-env/bin/activate # activate the environment
pip install -r requirements.txt # install required packages
```

## Documentation

In the manuscript, two deep learning models were evaluated for the segmentation of mSCTs: 3D U-Net and StarDist-3D. The 3D U-Net model is a fully convolutional neural network that is widely used for biomedical image segmentation. StarDist-3D is a deep learning model that combines a U-Net architecture with a star-convex polygon representation for the segmentation of objects with complex shapes. 

### Training and testing on your own calcium videos (Colab notebook)

The most simple way to train and test the models on your own calcium videos is to use the provided Colab notebook. The notebook is available [here](https://colab.research.google.com/github/FLClab/Calcium-Analysis/blob/main/CalciumUNet3D_ZeroCostDL4Mic.ipynb).

The notebook provides a step-by-step guide on how to train and test the 3D U-Net model on your own calcium videos. The notebook also provides a guide on how to use the pre-trained models to segment mSCTs in your own calcium videos.

### Training and testing on your own calcium videos (Advanced)

#### 3D U-Net

The training and prediction scripts of the 3D U-Net model are avaiable in the `experiments` folder. 

The steps to train the 3D U-Net model are as follows:
1. Train the model on the training dataset
```bash
python train.py --config configs/subset_0.25_1-0/MSCTS_UNet3D_subset_0.yml
```

2. Optimize the segmentation threshold
```bash
python optimize-segmentation.py --model <NAMEOFMODEL>
```

3. Predict the segmentation masks on the testing dataset
```bash
python predict.py --model <NAMEOFMODEL>
```

#### StarDist-3D

The training and prediction scripts of the StarDist-3D model are avaiable in the `baselines/StarDist3D` folder.

The steps to train the StarDist-3D model are as follows:
1. Train the model on the training dataset
```bash
python train.py --config configs/subset_0.25_1-0/MSCTS_StarDist3D_subset_0.yml
```

2. Optimize the segmentation threshold
```bash
python optimize.py --model <NAMEOFMODEL>
```

3. Predict the segmentation masks on the testing dataset
```bash
python predict.py --model <NAMEOFMODEL>
```

### Datasets

#### Training and validation datasets

The mSCT dataset is available for download [here](https://s3.valeria.science/flclab-calcium/index.html). The dataset is already divided into training and validation datasets. The dataset is provided in the form of an HDF5 file. The training and validation dataset are already pre-processed and normalized. 

The file architecture of the training dataset is as follows:
```
<file.hdf5>
|---train
|   |---<stream-id-0>
|   |   |---input: [N, H, W] (dtype: float32)
|   |   |---label: [N, H, W] (dtype: unit16)
|   |   |---events: [M, 7] (event-id, *bbox_coords)
|   |---<stream-id-1>
|   |   |---...
|   |---...
|---valid
|   |---<stream-id-0>
|   |   |---input: [N, H, W] (dtype: float32)
|   |   |---label: [N, H, W] (dtype: unit16)
|   |   |---events: [M, 7] (event-id, *bbox_coords)
|   |---<stream-id-1>
|   |   |---...
|   |---...    
```

#### Testing dataset

The testing dataset is also available for download [here](https://s3.valeria.science/flclab-calcium/index.html). The testing dataset is provided in the form of raw tifffile movies. The testing dataset is not pre-processed and normalized.

## Citation

If you use this code or the provided datasets, please cite the following paper:

```bibtex
@article{
    title={Quantitative Analysis of Miniature Synaptic Calcium Transients Using Positive Unlabeled Deep Learning}
}
```
