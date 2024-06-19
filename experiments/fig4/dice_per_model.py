import numpy as np
import matplotlib.pyplot as plt
import pickle
import tifffile
from model_paths import UNet_025, STARDIST_025
from tqdm import tqdm
import glob
import skimage.measure
import scipy
from metrics.segmentation import commons
from dice_vs_centroid import baseline, load_data, get_crop, get_deltaF
import pandas
import seaborn
import argparse 

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="UNet")
args = parser.parse_args()

best_1_0 = 2
best_25_0 = 23
best_25_1 = 23 
best_25_2 = 23
best_25_4 = 23
best_25_8 = 6
best_25_16 = 5
best_25_32 = 5
best_25_64 = 9
best_25_128 = 3
best_25_256 = 18
SD_4_0 = 4
SD_1_0 = 9
SD_1_1 = 2
SD_1_2 = 23
SD_1_4 = 16
SD_1_8 = 23
SD_1_16 = 8
SD_1_32 = 9
SD_1_64 = 6
SD_1_128 = 19
SD_1_256 = 5

MINIFINDER_PATH = "/home/frbea320/projects/def-flavielc/frbea320/mscts-analysis/data/testset/minifinder"
FLAVIE1_PATH = "/home/frbea320/projects/def-flavielc/frbea320/mscts-analysis/data/testset/segmentation-expert/mask_flavie_1"
FLAVIE2_PATH = "/home/frbea320/projects/def-flavielc/frbea320/mscts-analysis/data/testset/segmentation-expert/mask_flavie_2"
THERESA1_PATH = "/home/frbea320/projects/def-flavielc/frbea320/mscts-analysis/data/testset/segmentation-expert/mask_theresa_1"
THERESA2_PATH = "/home/frbea320/projects/def-flavielc/frbea320/mscts-analysis/data/testset/segmentation-expert/mask_theresa_2"
DATA_PATH = "/home/frbea320/projects/def-flavielc/frbea320/mscts-analysis/baselines/files"
RAW_PATH = "/home/frbea320/projects/def-flavielc/frbea320/mscts-analysis/data/testset/raw-input"

MODEL_PATHS = [
    f"/home/frbea320/projects/def-flavielc/frbea320/mscts-analysis/data/baselines/ZeroCostDL4Mic/{UNet_025['UNet3D_complete_1-0'][best_1_0]}",
    f"/home/frbea320/projects/def-flavielc/frbea320/mscts-analysis/data/baselines/ZeroCostDL4Mic/{UNet_025['25-0'][best_25_0]}",
    f"/home/frbea320/projects/def-flavielc/frbea320/mscts-analysis/data/baselines/ZeroCostDL4Mic/{UNet_025['25-1'][best_25_1]}",
    f"/home/frbea320/projects/def-flavielc/frbea320/mscts-analysis/data/baselines/ZeroCostDL4Mic/{UNet_025['25-2'][best_25_2]}",
    f"/home/frbea320/projects/def-flavielc/frbea320/mscts-analysis/data/baselines/ZeroCostDL4Mic/{UNet_025['25-4'][best_25_4]}",
    f"/home/frbea320/projects/def-flavielc/frbea320/mscts-analysis/data/baselines/ZeroCostDL4Mic/{UNet_025['25-8'][best_25_8]}",
    f"/home/frbea320/projects/def-flavielc/frbea320/mscts-analysis/data/baselines/ZeroCostDL4Mic/{UNet_025['25-16'][best_25_16]}",
    f"/home/frbea320/projects/def-flavielc/frbea320/mscts-analysis/data/baselines/ZeroCostDL4Mic/{UNet_025['25-32'][best_25_32]}",
    f"/home/frbea320/projects/def-flavielc/frbea320/mscts-analysis/data/baselines/ZeroCostDL4Mic/{UNet_025['25-64'][best_25_64]}",
    f"/home/frbea320/projects/def-flavielc/frbea320/mscts-analysis/data/baselines/ZeroCostDL4Mic/{UNet_025['25-128'][best_25_128]}",
    f"/home/frbea320/projects/def-flavielc/frbea320/mscts-analysis/data/baselines/ZeroCostDL4Mic/{UNet_025['25-256'][best_25_256]}"
]

STARDIST_PATHS = [
    f"/home/frbea320/scratch/StarDist3D/{STARDIST_025['4-0'][SD_4_0]}",
    f"/home/frbea320/projects/def-flavielc/frbea320/mscts-analysis/data/testset/StarDist3D/{STARDIST_025['1-0'][SD_1_0]}",
    f"/home/frbea320/projects/def-flavielc/frbea320/mscts-analysis/data/testset/StarDist3D/{STARDIST_025['1-1'][SD_1_1]}",
    f"/home/frbea320/projects/def-flavielc/frbea320/mscts-analysis/data/testset/StarDist3D/{STARDIST_025['1-2'][SD_1_2]}",
    f"/home/frbea320/projects/def-flavielc/frbea320/mscts-analysis/data/testset/StarDist3D/{STARDIST_025['1-4'][SD_1_4]}",
    f"/home/frbea320/projects/def-flavielc/frbea320/mscts-analysis/data/testset/StarDist3D/{STARDIST_025['1-8'][SD_1_8]}",
    f"/home/frbea320/projects/def-flavielc/frbea320/mscts-analysis/data/testset/StarDist3D/{STARDIST_025['1-16'][SD_1_16]}",
    f"/home/frbea320/projects/def-flavielc/frbea320/mscts-analysis/data/testset/StarDist3D/{STARDIST_025['1-32'][SD_1_32]}",
    f"/home/frbea320/projects/def-flavielc/frbea320/mscts-analysis/data/testset/StarDist3D/{STARDIST_025['1-64'][SD_1_64]}",
    f"/home/frbea320/projects/def-flavielc/frbea320/mscts-analysis/data/testset/StarDist3D/{STARDIST_025['1-128'][SD_1_128]}",
    f"/home/frbea320/projects/def-flavielc/frbea320/mscts-analysis/data/testset/StarDist3D/{STARDIST_025['1-256'][SD_1_256]}"
]

model_keys = ["4-0", "1-0", "1-1", "1-2", "1-4", "1-8", "1-16", "1-32", "1-64", "1-128", "1-256"]
event_keys = ["onSynapse", "onDendrite", "smallArea", "bigArea", "outOfFocus", "longArea", "highIntensity", "lowIntensity"]

def compute_unet_by_event_type(data: dict) -> np.ndarray:
    results = {
        event: {
            model: [] for model in model_keys
        } for event in event_keys
    }
    for event_type, values in tqdm(data.items(), desc="Event types..."):
        for movie_id, coords, time_slice, index in tqdm(values, desc="Events..."):
            movie = tifffile.imread(f"{RAW_PATH}/{movie_id}")
            truth_name = glob.glob(f"{THERESA2_PATH}/*-{index}.tif")[0]
            truth = tifffile.imread(truth_name)
            x, y, width, height = coords
            model_paths = [f"{p}/Quality Control Segmentation/Prediction/{movie_id}".replace(".tif", "_prediction.tif") for p in MODEL_PATHS]
            ###
            model_preds = [tifffile.imread(p) for p in model_paths]
            ###
            model_preds = [p[time_slice, y:y+height, x:x+width] for p in model_preds]
            model_die = [commons.dice(truth, p) for p in model_preds]
            for d, mkey in zip(model_die, model_keys):
                results[event_type][mkey].append(d)
    return results


def compute_stardist_by_event_type(data: dict) -> np.ndarray:
    results = {
        event: {
            model: [] for model in model_keys
        } for event in event_keys
    }
    for event_type, values in tqdm(data.items()):
        for movie_id, coords, time_slice, index in tqdm(values, desc="Events..."):
            movie = tifffile.imread(f"{RAW_PATH}/{movie_id}")
            truth_name = glob.glob(f"{THERESA2_PATH}/*-{index}.tif")[0]
            truth = tifffile.imread(truth_name)
            x, y, width, height = coords
            model_paths = [f"{p}/{movie_id}".replace(".tif", ".npz") for p in STARDIST_PATHS]
            model_die = []
            for p in model_paths:
                pred = np.load(p)['label']
                pred = pred[time_slice, y:y+height, x:x+width]
                dice = commons.dice(truth, pred)
                model_die.append(dice)
            for d, mkey in zip(model_die, model_keys):
                results[event_type][mkey].append(d)
    return results


def compare_stardist_to_expert(data: dict) -> np.ndarray:
    results = {
        "ITD": [],
        "4-0": [],
        "1-0": [],
        "1-1": [],
        "1-2": [],
        "1-4": [],
        "1-8": [],
        "1-16": [],
        "1-32": [],
        "1-64": [],
        "1-128": [],
        "1-256": [],
        "Expert 1": [],
        "Expert 2": []
    }
    for _, values in tqdm(data.items()):
        for movie_id, coords , time_slice, index in tqdm(values, desc="Events..."):
            movie = tifffile.imread(f"{RAW_PATH}/{movie_id}")
            truth_name = glob.glob(f"{FLAVIE1_PATH}/*-{index}.tif")[0]
            flavie_name = glob.glob(f"{FLAVIE2_PATH}/*-{index}.tif")[0]
            theresa_name = glob.glob(f"{THERESA1_PATH}/*-{index}.tif")
            truth = tifffile.imread(truth_name)
            flavie = tifffile.imread(flavie_name)
            theresa = tifffile.imread(theresa_name)
            x, y, width, height = coords
            model_paths = [f"{p}/{movie_id}".replace(".tif", ".npz") for p in STARDIST_PATHS]
            model_die = []
            for p in model_paths:
                pred = np.load(p)['label']
                pred = pred[time_slice, y:y+height, x:x+width]
                dice = commons.dice(truth, pred)
                model_die.append(dice)
            minifinder = np.load(f"{MINIFINDER_PATH}/{movie_id}".replace(".tif", ".npy"))
            minifinder = minifinder[time_slice, y:y+height, x:x+width]
            intra_dice = commons.dice(truth, flavie)
            inter_dice = commons.dice(truth, theresa)
            minifinder_dice = commons.dice(truth, minifinder)
            for d, key in zip(model_die, ["4-0", "1-0", "1-1", "1-2", "1-4", "1-8", "1-16", "1-32", "1-64", "1-128", "1-256"]):
                results[key].append(d)
            results["ITD"].append(minifinder_dice)
            results["Expert 1"].append(intra_dice)
            results["Expert 2"].append(inter_dice)
    return results


def compare_unet_to_expert(data: dict) -> np.ndarray:
    """
    Compares each UNet model, minifinder and one expert to an expert chosen as ground truth.

    Params:
    --------
        data: ground truth segmentation data
    
    Returns:
    --------
        results (dict): dice results
    """
    results = {
        "ITD": [],
        "4-0": [],
        "1-0": [],
        "1-1": [],
        "1-2": [],
        "1-4": [],
        "1-8": [],
        "1-16": [],
        "1-32": [],
        "1-64": [],
        "1-128": [],
        "1-256": [],
        "Expert 1": [],
        "Expert 2": [],
    }
    for event_type, values in tqdm(data.items(), desc="Event types..."):
        
        for movie_id, coords, time_slice, index in tqdm(values, desc="Events..."):
            movie = tifffile.imread(f"{RAW_PATH}/{movie_id}")
            truth_name = glob.glob(f"{FLAVIE1_PATH}/*-{index}.tif")[0]
            flavie_name = glob.glob(f"{FLAVIE2_PATH}/*-{index}.tif")[0]
            theresa_name = glob.glob(f"{THERESA1_PATH}/*-{index}.tif")
            truth = tifffile.imread(truth_name)
            flavie = tifffile.imread(flavie_name)
            theresa = tifffile.imread(theresa_name)
            x, y, width, height = coords
            movie_crop = movie[time_slice, y:y+height, x:x+width]
            ###
            model_paths = [f"{p}/Quality Control Segmentation/Prediction/{movie_id}".replace(".tif", "_prediction.tif") for p in MODEL_PATHS]
            ###
            model_preds = [tifffile.imread(p) for p in model_paths]
            ###
            model_preds = [p[time_slice, y:y+height, x:x+width] for p in model_preds]
            ###
            minifinder = np.load(f"{MINIFINDER_PATH}/{movie_id}".replace(".tif", ".npy"))
            minifinder = minifinder[time_slice, y:y+height, x:x+width]
            # populate dictionary
            model_die = [commons.dice(truth, p) for p in model_preds]
            intra_dice = commons.dice(truth, flavie)
            inter_dice = commons.dice(truth, theresa)
            minifinder_dice = commons.dice(truth, minifinder)
            for d, key in zip(model_die, ["4-0", "1-0", "1-1", "1-2", "1-4", "1-8", "1-16", "1-32", "1-64", "1-128", "1-256"]):
                results[key].append(d)
            results["ITD"].append(minifinder_dice)
            results["Expert 1"].append(intra_dice)
            results["Expert 2"].append(inter_dice)
    return results

def main():
    data = load_data()
    results = compute_unet_by_event_type(data=data) if args.model == "UNet" else compute_stardist_by_event_type(data=data)
    with open(f"./paper_figures/{args.model}_eventtype_theresa2_segmentation.pkl", "wb") as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)

if __name__=="__main__":
    main()
