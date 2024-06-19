import pickle
import numpy as np
import matplotlib.pyplot as plt
import glob
import skimage.measure
from metrics import CentroidDetectionError
from tqdm import tqdm
import argparse
import os
import pandas as pd
import tifffile
from collections import defaultdict
import utils

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True, type=str, help="Model directory containing the model predictions")
args = parser.parse_args()


GROUND_TRUTH_PATH = "/home/frbea320/projects/def-flavielc/frbea320/mscts-analysis/data/testset"
PREDICTION_PATH = f"/home/frbea320/projects/def-flavielc/frbea320/mscts-analysis/data/testset/StarDist3D/{args.model}"

manual_expert_files = glob.glob(os.path.join(GROUND_TRUTH_PATH, "manual-expert", "*.csv"))

def filter_rprops(regions, constraints):
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
#         filename = os.path.basename(file)
#         prediction_file = filename.split("_")[-1].split(".")[0]
#         prediction_file = os.path.join(PREDICTION_PATH, prediction_file + ".npz")
#         tail = os.path.splitext(os.path.basename(prediction_file))[0]
#         raw_pred = tifffile.imread(prediction_file.replace(".tif", "_prediction.tif"))    
#         movies[key + "_prediction"] = raw_pred   
    return movies

def init_keys():
    file = manual_expert_files[0]
    filename = os.path.basename(file)
    prediction_file = filename.split("_")[-1].split(".")[0]
    prediction_file = os.path.join(PREDICTION_PATH, prediction_file + ".npz")
    pred_data = np.load(prediction_file)
    label_keys = pred_data.files
    return label_keys

def save_dict(outdict, filename):
    with open(filename, "wb") as fp:
        pickle.dump(outdict, fp)
        print("Dictionary saved successfully to pickle file")
        
def get_detection_scores(label_key, movies):
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
        movie_name = filename.split("_")[-1].split(".")[0]
        prediction_file = os.path.join(PREDICTION_PATH, movie_name + ".npz")
        movie = movies[movie_name]
        pred_data = np.load(prediction_file, allow_pickle=True)
        print(f"\nFor file {prediction_file}")
        print(pred_data.files)
        try:
            raw_pred = pred_data[label_key]
        except:
            print(f"Prediction file {prediction_file} is problematic")
            continue
        pred_label = skimage.measure.label(raw_pred > 0)
        pred_rprops = skimage.measure.regionprops(pred_label, intensity_image=movie)
        pred_centroids = np.array([r.weighted_centroid for r in pred_rprops])
        detector = CentroidDetectionError(truth_centroids, pred_centroids, threshold=6, algorithm="hungarian")
        for key in scores.keys():
            scores[key].append(getattr(detector, key))
    return scores

def save_results(label_keys, movies):
    model_recall, model_precision = [], []
    results = defaultdict(list)
    for tau in tqdm(label_keys, desc="Thresholds..."):
        scores = get_detection_scores(label_key=tau, movies=movies)
        model_recall.append(np.mean(scores["recall"]))
        model_precision.append(np.mean(scores["precision"]))
        for key, values in scores.items():
            results[key].append(np.mean(values))
            results[key + "_per_video"].append(values)
    results["thresholds"] = label_keys
    return results
        
            
def main():
    label_keys = init_keys()
    label_keys.remove("label")
    movie_dict = preload_movies()
    results = save_results(
        model_name=args.model,
        label_keys=label_keys,
        movies=movie_dict,
    )
    save_dict(results, os.path.join(PREDICTION_PATH, "segmentation_results.pkl"))

if __name__=="__main__":
    main()
    
