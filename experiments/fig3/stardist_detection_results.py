import pickle
import numpy as np
import glob
import skimage.measure
from tqdm import tqdm
import argparse
import os
import pandas
import tifffile
from collections import defaultdict
from scipy.spatial import distance
from scipy.optimize import linear_sum_assignment

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True, type=str, help="Model directory containing the model predictions")
args = parser.parse_args()

GROUND_TRUTH_PATH = "/home/frbea320/projects/def-flavielc/frbea320/mscts-analysis/data/testset"
SD_SCRATCH_PATH = "/home/frbea320/scratch/StarDist3D"
SD_PREDICTIONS_PATH = "/home/frbea320/projects/def-flavielc/frbea320/mscts-analysis/data/testset/StarDist3D"
RAW_PATH = "/home/frbea320/projects/def-flavielc/frbea320/mscts-analysis/data/testset/raw-input"

thresholds = np.linspace(0.01, 0.99, 100)

POSTPROCESS_PARAMS = {
            'minimal_time': 3,
            'minimal_height': 3,
            'minimal_width': 3,
        }

    
manual_expert_files = glob.glob(os.path.join(GROUND_TRUTH_PATH, "manual-expert", "*.csv"))

def get_movie(fname):
    movie = tifffile.imread(f"{RAW_PATH}/{fname}.tif")
    return movie

def compute_cost_matrix(truth, pred):
    if (len(truth) < 1) or ((len(pred)) < 1):
        cost_matrix = np.ones((len(truth), len(pred))) * 1e+6
    else:
        cost_matrix = distance.cdist(truth, pred, metric='euclidean')
    return cost_matrix


def assign_hungarian(cost_matrix, threshold=6):
    truth_indices = np.arange(cost_matrix.shape[0])
    pred_indices = np.arange(cost_matrix.shape[1])
    false_positives = np.sum(cost_matrix < threshold, axis=0) == 0
    false_negatives = np.sum(cost_matrix < threshold, axis=1) == 0
    cost = cost_matrix[~false_negatives][:, ~false_positives]
    truth_indices = truth_indices[~false_negatives]
    pred_indices = pred_indices[~false_positives]
    truth_couple, pred_couple = linear_sum_assignment(np.log(cost + 1e-6), maximize=False)
    distances = cost[truth_couple, pred_couple]
    truth_couple = truth_couple[distances < threshold]
    pred_couple = pred_couple[distances < threshold]
    truth_couple = truth_indices[truth_couple]
    pred_couple = pred_indices[pred_couple]
    return truth_couple, pred_couple

def get_fp(cost_matrix, pred_couple):
    if cost_matrix.shape[1] > 0:
        return np.array(list(set(range(cost_matrix.shape[1])) - set(pred_couple)))
    else:
        return np.array([])
    
def get_fn(cost_matrix, truth_couple):
    if cost_matrix.shape[0] > 0:
        return np.array(list(set(range(cost_matrix.shape[0])) - set(truth_couple)))
    else:
        return np.array([])
    

def get_precision(tp: int, fp: int):
    epsilon = 1e-5 if tp + fp == 0 else 0
    return tp / (tp + fp + epsilon)

def get_recall(tp: int, fn: int):
    epsilon = 1e-5 if tp + fn == 0 else 0
    return tp / (tp+fn + epsilon)

def get_f1(recall: float, precision: float):
    epsilon = 1e-5 if precision + recall == 0 else 0
    return (2*precision * recall) / (precision+recall+epsilon)

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


def save_dict(outdict: dict, filename: str) -> None:
    with open(filename, "wb") as fp:
        pickle.dump(outdict, fp)
        print(f"\n ***** Saving result dictionary to {args.model} *****")

def compute_cornichons() -> None:
    if os.path.exists(f"{args.model}/results.pkl"):
        print("Results have already been generated, skipping...")
        return
    num_videos = len(manual_expert_files)
    scores = {
        "recall": np.zeros((num_videos, 100)),
        "precision": np.zeros((num_videos, 100)),
        "true_positives": np.zeros((num_videos, 100)),
        "false_positives": np.zeros((num_videos, 100)),
        "false_negatives": np.zeros((num_videos, 100))
    }
    for i, file in enumerate(tqdm(manual_expert_files, desc="Videos...")):
        data = pandas.read_csv(file)
        truth_centroids = np.stack((data["Slice"], data["Y"], data["X"]), axis=-1)
        path = args.model  
        fname = file.split("/")[-1].split(".")[0].split("_")[-1]
        movie = get_movie(fname)
        try:
            prediction_path = f"{path}/{fname}.npz"
        except FileNotFoundError:
            temp_path = path.split("/")[-1]
            print(f"{fname} does not exist in {temp_path}")
            continue
        prediction = np.load(prediction_path)
        thresholds = prediction.files
        thresholds.remove("label")
        for j, tau in enumerate(thresholds):
            pred = prediction[tau]
            pred_label = skimage.measure.label(pred)
            pred_rprops = skimage.measure.regionprops(pred_label, intensity_image=movie)
            pred_rprops = filter_rprops(pred_rprops, constraints=POSTPROCESS_PARAMS)
            pred_centroids = np.array([r.weighted_centroid for r in pred_rprops])
            cost_matrix = compute_cost_matrix(truth_centroids, pred_centroids)
            truth_couple, pred_couple = assign_hungarian(cost_matrix)
            num_tp = pred_couple.shape[0]
            num_fp = get_fp(cost_matrix, pred_couple).shape[0]
            num_fn = get_fn(cost_matrix, truth_couple).shape[0]
            precision = get_precision(num_tp, num_fp)
            recall = get_recall(num_tp, num_fn)
            scores["precision"][i][j] = precision
            scores["recall"][i][j] = recall
            scores["true_positives"][i][j] = num_tp
            scores["false_positives"][i][j] = num_fp
            scores["false_negatives"][i][j] = num_fn
    scores["precision"] = np.mean(scores["precision"], axis=0)
    scores["recall"] = np.mean(scores["recall"], axis=0)
    scores["true_positives"] = np.mean(scores["true_positives"], axis=0)
    scores["false_positives"] = np.mean(scores["false_positives"], axis=0)
    scores["false_negatives"] = np.mean(scores["false_negatives"], axis=0)
    save_dict(scores, filename=f"{args.model}/results.pkl")

def main():
    compute_cornichons()

if __name__=="__main__":
    main()


