import scipy
import operator
import numpy as np
import matplotlib.pyplot as plt
import json
import glob
import skimage.measure
from metrics import commons
from tqdm import tqdm
import argparse
import os
import utils
import pandas as pd
import scipy.stats
import tifffile
import sys
sys.path.insert(0, "../fig3")
from detection_results import filter_rprops, preload_movies, save_dict
from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True, type=str,
                    help="Directory containing the model's predictions")
parser.add_argument("--expert", required=False,
                    type=str, default="mask_flavie_1")
args = parser.parse_args()

GROUND_TRUTH_PATH = "../../data/testset"
PREDICTIONS_PATH = f"../../data/testset/UNet3D/{args.model}"

thresholds = np.linspace(0.01, 0.99, 100)


POSTPROCESS_PARAMS = {
    'minimal_time': 3,
    'minimal_height': 3,
    'minimal_width': 3,
}

results = {
    "dice": [],
}


def load_data():
    with open("../../baselines/files/macro.ijm", "r") as file:
        lines = [
            line.rstrip() for line in file.readlines()
            if (line.startswith("eventType")) or
               (line.startswith("open(rawFolder")) or
               (line.startswith("makeRectangle(")) or
               (line.startswith("duplicateAndSave("))
        ]

    out = defaultdict(list)
    key = None
    i = 1
    flag = False
    for line in lines:
        if line.startswith("eventType"):
            key = line.split('"')[-2]
            continue
        if line.startswith("open("):
            line = line.split('"')[-2]
        if line.startswith("makeRectangle("):
            line = eval(line.split("makeRectangle")[-1][:-1])
        if isinstance(line, str) and line.startswith("duplicateAndSave("):
            flag = True
            line = eval(line.split("duplicateAndSave(")[-1].split(",")[0])
        out[key].append(line)
        if flag:
            out[key].append(i)
            i += 1
            flag = False
    for key, values in out.items():
        new_values = []
        tmp = []
        for i, value in enumerate(values):
            if (i > 0) and (i % 4 == 0):
                new_values.append(tmp)
                tmp = []
            tmp.append(value)
        out[key] = new_values
    return out


def get_crop(array, location, bounding, pad=False, **kwargs):
    """
    Permet de retourner les coordonnés du pour faire le crop dans l'array donnée.
    Si les dimensions du crop dépassent celles de l'array, le crop sera plus petit.
    INPUT:
        array: array quelconque
        location: spécifie le centre du crop
        size: si un seul chiffre, ce chiffre est utilisé comme longueur pour chaque dimension,
              sinon size doit avoir la même dimension que array
        pad: si pad est True, on pad le array pour avoir la bonne size
    OUPUT:
        retourne les coordonnées pour effectuer le crop dans l'array
    """
    arrayShape = array.shape

    if len(bounding) != len(arrayShape):
        ValueError("The size of the crop should be the same size as the array")

    start = tuple(map(lambda a, da: (np.round(a) - da //
                  2).astype(int), location, bounding))
    end = tuple(map(operator.add, start, bounding))
    if pad:
        padNumpy = []
        for low, up, arrShape in zip(start, end, arrayShape):
            pad_min = -low if low < 0 else 0
            pad_max = up - arrShape if (up - arrShape) > 0 else 0
            padNumpy.append((pad_min, pad_max))
        padNumpy = tuple(padNumpy)

    start = tuple(map(lambda a: np.clip(a, 0, None), start))
    end = tuple(map(lambda a, dmax: np.clip(
        a, 0, dmax).astype(int), end, array.shape))
    slices = tuple(map(slice, start, end))
    array = array[slices]

    if pad:
        array = np.pad(array, padNumpy, **kwargs)

    return array, slices


def baseline(y, lam=1e3, ratio=1e-6):
    """
    Provient de https://github.com/charlesll/rampy/blob/master/rampy/baseline.py
    """
    N = len(y)
    D = scipy.sparse.csc_matrix(np.diff(np.eye(N), 2))
    w = np.ones(N)
    MAX_ITER = 100

    for _ in range(MAX_ITER):
        W = scipy.sparse.spdiags(w, 0, N, N)
        Z = W + lam * D.dot(D.transpose())
        z = scipy.sparse.linalg.spsolve(Z, w * y)
        d = y - z
        # make d- and get w^t with m and s
        dn = d[d < 0]
        m = np.mean(dn)
        s = np.std(dn)
        wt = 1.0 / (1 + np.exp(2 * (d - (2 * s - m)) / s))
        # check exit condition and backup
        if np.linalg.norm(w - wt) / np.linalg.norm(w) < ratio:
            break
        w = wt

    return z


def get_deltaF(image, center):
    _, slices = get_crop(image, center, (1, 5, 5))
    trace = np.mean(image[:, slices[1], slices[2]], axis=(1, 2))

    # Compute normalized trace
    F0 = baseline(trace, lam=1e6, ratio=1e-10)

    trace_normalized = (trace - F0) / F0
    deltaF_frame = np.round(center[0]).astype(int)
    deltaF = trace_normalized[deltaF_frame]
    return deltaF


def get_segmentation_scores(data, tau):
    scores = {
        "dice": [],
        "is_empty": []
    }
    for event_type, values in tqdm(data.items(), desc="Event types...", leave=False):
        for movie_id, coords, time_slice, index in tqdm(values, desc="Events..."):
            movie = tifffile.imread(os.path.join(
                GROUND_TRUTH_PATH, "raw-input", movie_id))
            truth_name = glob.glob(os.path.join(
                GROUND_TRUTH_PATH, "segmentation-expert", args.expert, f"*-{index}.tif"))[0]
            truth = tifffile.imread(truth_name)
            x, y, width, height = coords

            pred_path = os.path.join(
                PREDICTIONS_PATH, "raw-input", movie_id.replace(".tif", "_prediction.tif"))
            raw_pred = tifffile.imread(pred_path)
            pred = raw_pred > tau
            pred = pred[time_slice, y:y + height, x:x + width]
            deltaF = get_deltaF(movie, np.array(
                [time_slice, int(y + height / 2), int(x + width / 2)]))
            # empty predictions are penalized in the detection script, not here
            is_empty = np.count_nonzero(pred) == 0

            diceval = commons.dice(truth, pred)
            scores["dice"].append(diceval)
            scores["is_empty"].append(is_empty)
    return scores


def save_results(data):
    model_dice = []
    model_scores = []
    for tau in tqdm(thresholds, desc="Thresholds..."):
        scores = get_segmentation_scores(data, tau)
        model_dice.append(np.mean(scores["dice"]))
        model_scores.append(scores)
    results["dice"] = model_dice
    results["all"] = model_scores
    results["thresholds"] = thresholds

def main():
    data = load_data()
    save_results(data)
    save_dict(results, os.path.join(
        PREDICTIONS_PATH, "segmentation_results.pkl"))


if __name__ == "__main__":
    main()
