import pickle
import glob 
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
sys.path.insert(0, "..")
from tqdm import tqdm
import tifffile
import skimage.measure 
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, required=True, help="Name of the model to load")
args = parser.parse_args()

GROUND_TRUTH_PATH = "/home/frbea320/projects/def-flavielc/frbea320/mscts-analysis/data/testset"
SD_SCRATCH_PATH = "/home/frbea320/scratch/StarDist3D"
SD_PREDICTIONS_PATH = "/home/frbea320/projects/def-flavielc/frbea320/mscts-analysis/data/testset/StarDist3D"
RAW_PATH = "/home/frbea320/projects/def-flavielc/frbea320/mscts-analysis/data/testset/raw-input"

random_seed = 42

manual_expert_files = glob.glob(os.path.join(GROUND_TRUTH_PATH, "manual-expert", "*.csv"))

POSTPROCESS_PARAMS = {
            'minimal_time': 3,
            'minimal_height': 3,
            'minimal_width': 3,
        }

PROPERTIES=[
    "label", 
    "area", 
    "major_axis_length", 
    "minor_axis_length", 
    "bbox", 
    "centroid", 
    "max_intensity", 
    "min_intensity",
    # "moments",
    # "solidity"
]


def get_movie(fname):
    movie = tifffile.imread(f"{RAW_PATH}/{fname}.tif")
    return movie

def filter_rprops(regions: list, constraints: dict):
    """
    Enleve les événements selon trop petit selon différents critères
    """

    filtered_regions = []
    for region in regions:
        area = region.area
        center = np.array(region.centroid)

        t1, h1, w1, t2, h2, w2 = region.bbox
        lenT = t2 - t1
        lenH = h2 - h1
        lenW = w2 - w1

        # Constraints check
        good = True
        if 'minimal_time' in constraints.keys():
            if lenT < constraints['minimal_time']:
                good = False
        if 'minimal_height' in constraints.keys():
            if lenH < constraints['minimal_height']:
                good = False
        if 'minimal_width' in constraints.keys():
            if lenW < constraints['minimal_width']:
                good = False
        if 'maximal_time' in constraints.keys():
            if lenT > constraints['maximal_time']:
                good = False
        if good:
            filtered_regions.append(region)
    return filtered_regions

def get_segmentation_regionprops():
    out = {}
    for i, file in enumerate(tqdm(manual_expert_files, desc="Videos...")):
        path = args.model  
        fname = file.split("/")[-1].split(".")[0].split("_")[-1] # e.g. 19-2
        movie = get_movie(fname)
        prediction_path = f"{path}/{fname}.npz"
        try:
            prediction = np.load(prediction_path)['labels']
        except:
            prediction = np.load(prediction_path)['label']
        pred_label = skimage.measure.label(prediction)
        try:
            pred_rprops = skimage.measure.regionprops_table(pred_label, intensity_image=movie, properties=PROPERTIES)
        except ValueError:
            print("Caught Value Error... Skipping file;")
            continue
        out[fname] = pred_rprops
    return out
    
def main():
    segmentation_rprops = get_segmentation_regionprops()
    with open(f"{args.model}/regionprops.pkl", "wb") as fp:
        pickle.dump(segmentation_rprops, fp)
        print("Segmentation dictionary saved successfully to pickle file")  

if __name__=="__main__":
    main()