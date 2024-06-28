from __future__ import print_function, unicode_literals, absolute_import, division
import sys
import numpy
import matplotlib
import random
# matplotlib.rcParams["image.interpolation"] = None
from matplotlib import pyplot
import tensorflow
import yaml
import datetime
import uuid
import os

from glob import glob
from tqdm import tqdm
from tifffile import imread
from csbdeep.utils import Path, normalize
from tensorflow.keras.utils import Sequence
from skimage import measure

from stardist import fill_label_holes, random_label_cmap, calculate_extents, gputools_available
from stardist import Rays_GoldenSpiral
from stardist.matching import matching, matching_dataset
from stardist.models import Config3D, StarDist3D, StarDistData3D

import loader

numpy.random.seed(42)
lbl_cmap = random_label_cmap()

def random_flip(image, mask):
    """
    Implements a random flip of the image and mask

    :param image: A `numpy.ndarray` of the image
    :param mask: A `numpy.ndarray` of the mask

    :returns : A `numpy.ndarray` of the transformed image
               A `numpy.ndarray` of the transformed mask
    """
    possible_flips = (1, 2, None) # 1 flipud, 2 fliplr, None nothing
    flipping_mode = random.choice(possible_flips)
    if flipping_mode:
        image = numpy.flip(image, flipping_mode)
        mask = numpy.flip(mask, flipping_mode)
    return image, mask

def random_crop(image, mask, widths=[32, 32, 32]):
    """
    Implements a random crop of the image and mask

    :param image: A `numpy.ndarray` of the image
    :param mask: A `numpy.ndarray` of the mask

    :returns : A `numpy.ndarray` of the transformed image
               A `numpy.ndarray` of the transformed mask
    """
    corners = numpy.random.randint([w - 1 for w in widths])
    slc = tuple(
        slice(c, c + w) for c, w in zip(corners, widths)
    )
    return image[slc], mask[slc]

def augmenter(x, y):
    """Augmentation of a single input/label image pair.
    x is an input image
    y is the corresponding ground-truth label image
    """
    # Random crop is done via stardist
    # x, y = random_crop(x, y)
    x, y = random_flip(x, y)
    return x, y

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True,
                        help="Specify the path of the config file")
    parser.add_argument("--dry-run", action="store_true",
                        help="Performs a dry-run")
    parser.add_argument("--seed", required=False, default=None, type=int,
                    help="(optional) Update the random seed during training")

    args = parser.parse_args()

    gpus = tensorflow.config.list_physical_devices('GPU')
    if gpus:
      # Restrict TensorFlow to only allocate XGB of memory on the first GPU
      try:
        tensorflow.config.set_logical_device_configuration(
            gpus[0],
            [tensorflow.config.LogicalDeviceConfiguration(memory_limit=8 * 1024)])
        logical_gpus = tensorflow.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
      except RuntimeError as e:
        # Virtual devices must be set before GPUs have been initialized
        print(e)

    config = yaml.load(open(args.config, "r"), Loader=yaml.Loader)

    # Sets random seed
    seed = args.seed
    random.seed(seed)
    numpy.random.seed(seed)
    tensorflow.random.set_seed(seed)

    config["random-state"] = seed

    # Creates Sequence object for training
    dataset_path = "./data/80-20_calcium_dataset.h5"
    train_input_msct_sequence = loader.MSCTSequence(
        dataset_path, folds=config["train-folds"], label="input",
        samples_pu=config["samples_pu"],
        max_cache_size=32e+9, cache_mode="full"
    )
    train_label_msct_sequence = loader.MSCTSequence(
        dataset_path, folds=config["train-folds"], label="label",
        samples_pu=config["samples_pu"],
        max_cache_size=32e+9, cache_mode="full"
    )

    valid_input_msct_sequence = loader.MSCTSequence(
        dataset_path, folds=config["valid-folds"], label="input",
        samples_pu=config["samples_pu"],
        max_cache_size=10e+9, cache_mode="normal"
    )
    valid_label_msct_sequence = loader.MSCTSequence(
        dataset_path, folds=config["valid-folds"], label="label",
        samples_pu=config["samples_pu"],
        max_cache_size=10e+9, cache_mode="normal"
    )

    # Y = [train_label_msct_sequence[i] for i in numpy.random.choice(len(train_label_msct_sequence), size=1000, replace=False)]
    # extents = calculate_extents(Y)
    # anisotropy = tuple(numpy.max(extents) / extents)

    anisotropy = (1.7, 1.08, 1.0)
    print('empirical anisotropy of labeled objects = %s' % str(anisotropy))

    # 96 is a good default choice (see 1_data.ipynb)
    n_rays = 96
    n_channel = 1

    # Use OpenCL-based computations for data generator during training (requires 'gputools')
    use_gpu = False and gputools_available()

    # Predict on subsampled grid for increased efficiency and larger field of view
    grid = tuple(1 if a > 1.5 else 2 for a in anisotropy)

    # Use rays on a Fibonacci lattice adjusted for measured anisotropy of the training data
    rays = Rays_GoldenSpiral(n_rays, anisotropy=anisotropy)

    conf = Config3D (
        rays             = rays,
        grid             = grid,
        anisotropy       = anisotropy,
        use_gpu          = use_gpu,
        n_channel_in     = n_channel,
        # adjust for your data below (make patch size as large as possible)
        train_tensorboard = False,
        train_patch_size = (32,32,32),
        train_batch_size = 64,
        train_sample_cache = False,
        train_steps_per_epoch = 100,
        train_epochs = 2 if args.dry_run else 400,
    )

    now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    unique_id = str(uuid.uuid4())[:8]
    name = "_".join((now, unique_id, config["model-name"], str(seed)))
    if args.dry_run:
        name = "debug"
    basedir = config.get("basedir", './data/baselines/StarDist3D/models')
    model = StarDist3D(conf, name=name, basedir=basedir)

    # Saves config file used for training
    yaml.dump(config, open(os.path.join(basedir, name, "config.yml"), "w"))

    history = model.train(
        train_input_msct_sequence, train_label_msct_sequence,
        validation_data = (valid_input_msct_sequence, valid_label_msct_sequence),
        augmenter = augmenter,
        seed = seed,
        workers = 4
    )
