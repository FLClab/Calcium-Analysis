import pickle
import numpy as np
import matplotlib.pyplot as plt
import json
import glob
import skimage.measure
from metrics import CentroidDetectionError
from tqdm import tqdm
import argparse
import os
import utils 
import pandas as pd
import scipy.stats
import tifffile
from collections import defaultdict


########################### SCRIPT ARGUMENTS ###########################
parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True, type=str, help="Model directory containing the model predictions")
args = parser.parse_args()

########################### GLOBAL VARIABLES ###########################

GROUND_TRUTH_PATH = "../../data/testset"
PREDICTIONS_PATH = f"../../data/testset/UNet3D/{args.model}"

thresholds = np.linspace(0.01, 0.99, 100)

POSTPROCESS_PARAMS = {
            'minimal_time': 3,
            'minimal_height': 3,
            'minimal_width': 3,
        }

    
manual_expert_files = glob.glob(os.path.join(GROUND_TRUTH_PATH, "manual-expert", "*.csv"))

# results = {
#     "recall": [],
#     "precision": []
# }
    
########################### METHODS ###########################
    
def filter_rprops(regions, constraints):
    """
    Enleve les événements selon trop petit selon différents critères
    """
    regionRemovedCount = 0
    pbar = tqdm(regions, total=len(regions), leave=False)
    filtered_regions = []
    for region in pbar:
        area = region.area
        center = np.array(region.centroid)

        t1, h1, w1, t2, h2, w2 = region.bbox
        lenT = t2 - t1
        lenH = h2 - h1
        lenW = w2 - w1

        # Constraints check
        good = True
        stateStr = ''
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
        else:
            regionRemovedCount += 1
        pbar.set_description(f'{regionRemovedCount} Regions removed so far')
    return filtered_regions


def preload_movies():
    movies = {}
    for file in tqdm(manual_expert_files, desc="Loading movies..."):
        # Raw movie
        filename = os.path.basename(file)
        prediction_file = filename.split("_")[-1].split(".")[0]
        key = str(prediction_file)
        tail = prediction_file + ".tif"
        input_file = f"{GROUND_TRUTH_PATH}/raw-input/{tail}"
        movie = tifffile.imread(input_file)
        movies[key] = movie
        
        # Prediction movie
        filename = os.path.basename(file)
        prediction_file = filename.split("_")[-1].split(".")[0]
        prediction_file = os.path.join(PREDICTIONS_PATH, "raw-input", prediction_file + ".tif")
        tail = os.path.splitext(os.path.basename(prediction_file))[0]
        
        raw_pred = tifffile.imread(prediction_file.replace(".tif", "_prediction.tif"))
        
        movies[key + "_prediction"] = raw_pred
        
    return movies

def save_dict(outdict, filename):
    with open(filename, "wb") as fp:
        pickle.dump(outdict, fp)
        print("Dictionary saved successfully to pickle file")
        
        
def get_detection_scores(model_name, tau, movies):
    scores = {
        "recall": [],
        "precision": [],
        "false_positive": [],
        "false_negative": [],
        "true_positive": [],
    }
    for file in tqdm(manual_expert_files, desc="Videos..."):
        data = pd.read_csv(file)
        truth_centroids = np.stack((data["Slice"], data["Y"], data["X"]), axis=-1)
        filename = os.path.basename(file)
        prediction_file = filename.split("_")[-1].split(".")[0]
        prediction_file = os.path.join(PREDICTIONS_PATH, prediction_file + ".tif")
        tail = os.path.splitext(os.path.basename(prediction_file))[0]

        movie = movies[tail]
        raw_pred = movies[tail + "_prediction"]
        # raw_pred = tifffile.imread(prediction_file.replace(".tif", "_prediction.tif"))
        
        prediction = raw_pred > tau
        pred_label = skimage.measure.label(prediction)
        pred_rprops = skimage.measure.regionprops(pred_label, intensity_image=movie)
        pred_rprops = filter_rprops(pred_rprops, POSTPROCESS_PARAMS)
        pred_centroids = np.array([r.weighted_centroid for r in pred_rprops])
        detector = CentroidDetectionError(truth_centroids, pred_centroids, threshold=6, algorithm="hungarian")
        for key in scores.keys():
            scores[key].append(getattr(detector, key))
    return scores

def save_results(model_name, movies):
    model_recall, model_precision = [], []
    results = defaultdict(list)
    for tau in tqdm(thresholds, "Thresholds..."):
        scores = get_detection_scores(model_name, tau=tau, movies=movies)
        model_recall.append(np.mean(scores["recall"]))
        model_precision.append(np.mean(scores["precision"]))
        
        for key, values in scores.items():
            results[key].append(np.mean(values))
            results[key + "_per_video"].append(values)

    results["thresholds"] = thresholds
    return results
                     
                     
########################### __MAIN__ ###########################
def main():
    movie_dict = preload_movies()
    results = save_results(
        model_name=args.model,
        movies=movie_dict,
    )
    save_dict(results, os.path.join(PREDICTIONS_PATH, "results.pkl"))
    
if __name__=="__main__":
    main()