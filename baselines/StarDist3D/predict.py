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
import tifffile

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
    parser.add_argument("--model", type=str, required=True,
                        help="Specify the model name")
    parser.add_argument("--basedir", type=str, default='./data/baselines/StarDist3D/models',
                        help="Specify the path of the config file")
    parser.add_argument("--dry-run", action="store_true",
                        help="Performs a dry-run")
    parser.add_argument("--variable-thresholds", action="store_true",
                        help="Allows to sequentially predict on the same video with different thresholds")
    parser.add_argument("--overwrite", action="store_true",
                        help="Whether images should be overwritten")    
    parser.add_argument("--data-folder", type=str, default=None, help="Name of the model to load")
    parser.add_argument("--result-folder", type=str, default=None, help="Name of the model to load")    
    parser.add_argument("--save-tiff", action="store_true", 
                        help="Whether to save as tiff files")        

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

    if os.path.isfile(args.model):
        with open(args.model, "r") as file:
            model_names = [line.rstrip() for line in file.readlines()]
    else:
        model_names = [args.model]

    for model_name in model_names:

        config = yaml.load(open(os.path.join(args.basedir, model_name, "config.yml"), "r"), Loader=yaml.Loader)
        if "test_data" not in config:
            config["test-data"] = "./data/testset"
        if isinstance(args.data_folder, str):
            config["test-data"] = args.data_folder

        thresholds = json.load(open(os.path.join(args.basedir, model_name, "thresholds.json"), "r"))

        if args.save_tiff:
            out_name = model_name
        else:
            if "samples_pu" in config:
                samples_pu = config["samples_pu"]
                out_name = "StarDist3D_{}_1-{}_{}".format(
                    samples_pu["positive"], samples_pu["unlabeled"],
                    model_name.split("_")[-1]
                )
            else:
                out_name = "StarDist3D"

        # Sets random seed
        seed = config.get("random-state", None)
        random.seed(seed)
        numpy.random.seed(seed)
        tensorflow.random.set_seed(seed)

        model = StarDist3D(None, name=model_name, basedir=args.basedir)
        model.thresholds = thresholds

        available_thresholds = {
            f"labels-{round(p, 3):0.3f}" : {"nms" : thresholds["nms"], "prob" : p} for p in numpy.linspace(0.01, 0.99, 100)
        }

        if args.save_tiff:
            dataset = loader.FolderMSCTSequence(**config)
        else:
            dataset = loader.TestMSCTSequence(**config)            

        for name, image, regionprops in tqdm(dataset, desc="Images"):
            # cz, cy, cx = map(int, regionprops[len(regionprops) // 2].centroid)
            
            if not args.overwrite and os.path.isfile(os.path.join(config["test-data"], out_name, name.replace(".tif", ".npz"))):
                continue

            model.thresholds = thresholds

            if args.save_tiff:

                result_folder = args.result_folder
                data_folder = args.data_folder
                source_file = name

                experiment_name = os.path.basename(data_folder)
                savename = source_file.split(data_folder)[-1]
                if savename.startswith(os.path.sep):
                    savename = savename[1:]
                dirname = os.path.dirname(savename)
                savename = os.path.splitext(os.path.basename(source_file))[0] + "_prediction.tif"        
                
                savename = os.path.join(result_folder, experiment_name, dirname, savename)

                if not args.overwrite and os.path.isfile(savename):
                    print(f"Skipping: {savename}")
                    continue

                labels, _ = model.predict_instances(
                    image, axes="ZYX",  n_tiles=(4, 4, 4))
            
                os.makedirs(os.path.dirname(savename), exist_ok=True)
                tifffile.imwrite(savename, labels.astype(numpy.uint16))

                del labels
            else:

                prob, dist = model.predict(
                    image, axes="ZYX", n_tiles=(4, 4, 4),
                    show_tile_progress=True,
                )
                points = None

                axes = "ZYX"
                _axes         = model._normalize_axes(image, axes)
                _axes_net     = model.config.axes
                _permute_axes = model._make_permute_axes(_axes, _axes_net)
                _shape_inst   = tuple(s for s,a in zip(_permute_axes(image).shape, _axes_net) if a != 'C')

                masks, _ = model._instances_from_prediction(_shape_inst, prob, dist,
                                                                points=points,
                                                                prob_class=None,
                                                                prob_thresh=None,
                                                                nms_thresh=None,
                                                                return_labels=True,
                                                                overlap_label=None,
                                                                **{})

                labels = {}
                labels["label"] = masks

                if args.variable_thresholds:
                    for key, thresh in tqdm(available_thresholds.items(), desc="PR-thresholds", leave=False):
                        model.thresholds = thresh
                        masks, _ = model._instances_from_prediction(_shape_inst, prob, dist,
                                                                        points=points,
                                                                        prob_class=None,
                                                                        prob_thresh=None,
                                                                        nms_thresh=None,
                                                                        return_labels=True,
                                                                        overlap_label=None,
                                                                        **{})
                        labels[key] = masks

                os.makedirs(os.path.join(
                    config["test-data"], "StarDist3D", out_name
                ), exist_ok=True)

                numpy.savez_compressed(
                    os.path.join(config["test-data"], "StarDist3D", out_name, name.replace(".tif", ".npz")),
                    **labels
                )
                
                del labels, prob, dist, masks, image, regionprops