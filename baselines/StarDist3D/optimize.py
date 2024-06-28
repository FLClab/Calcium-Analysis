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
import json
import pandas

from glob import glob
from tqdm import tqdm
from tifffile import imread
from csbdeep.utils import Path, normalize
from tensorflow.keras.utils import Sequence
from skimage import measure
from collections import defaultdict

from stardist import fill_label_holes, random_label_cmap, calculate_extents, gputools_available
from stardist import Rays_GoldenSpiral
from stardist.matching import matching, matching_dataset
from stardist.models import Config3D, StarDist3D, StarDistData3D

import loader

numpy.random.seed(42)
lbl_cmap = random_label_cmap()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--basedir", type=str, default='./data/baselines/StarDist3D/models',
                        help="Specify the path of the config file")
    parser.add_argument("--repetitions", type=int, default=5,
                        help="Specify the number of repetitions of samples")
    parser.add_argument("--n-samples", type=int, default=300,
                        help="Specify the number of repetitions of samples")
    parser.add_argument("--model", type=str, default=None,
                        help="Specify the name of a model to optimize OR a text file containing the name of multiple models")
    parser.add_argument("--dry-run", action="store_true",
                        help="Performs a dry-run")

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

    dataset_path = "./data/80-20_calcium_dataset.h5"

    if os.path.isfile(args.model):
        with open(args.model, "r") as file:
            model_names = [line.rstrip() for line in file.readlines()]
    else:
        model_names = [args.model]

    for model_name in model_names:
        config = yaml.load(open(os.path.join(args.basedir, model_name, "config.yml"), "r"), Loader=yaml.Loader)

        # Sets random seed
        seed = config.get("random-state", None)
        random.seed(seed)
        numpy.random.seed(seed)
        tensorflow.random.set_seed(seed)

        thresholds = defaultdict(list)
        for _ in range(args.repetitions):
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

            choices = numpy.random.choice(len(valid_input_msct_sequence), size=args.n_samples, replace=False)
            model = StarDist3D(None, name=model_name, basedir=args.basedir)
            out = model.optimize_thresholds(valid_input_msct_sequence[choices], valid_label_msct_sequence[choices])
            for key, value in out.items():
                thresholds[key].append(value)

        thresholds = {
            key : float(numpy.mean(values)) for key, values in thresholds.items()
        }
        # Saves thresholds
        json.dump(thresholds, open(os.path.join(args.basedir, model_name, "thresholds.json"), "w"))

        # df = pandas.read_csv(os.path.join(args.basedir, model_name, "logs.csv"))
        # print(df["val_loss"])
