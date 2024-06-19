import numpy as np
import matplotlib.pyplot as plt
import pickle
import tifffile
from model_paths import UNet_025
from collections import defaultdict
import operator
from tqdm import tqdm
import glob
import skimage.measure
import scipy
from metrics.segmentation import commons
from metrics import CentroidDetectionError
from scipy.spatial.distance import cdist
from itertools import combinations


BASE_PATH = f"/home/frbea320/projects/def-flavielc/frbea320/mscts-analysis/data/baselines/ZeroCostDL4Mic"
RAW_PATH = "/home/frbea320/projects/def-flavielc/frbea320/mscts-analysis/data/testset/raw-input"
EXPERT_PATH = "/home/frbea320/projects/def-flavielc/frbea320/mscts-analysis/data/testset/segmentation-expert/mask_flavie_1"
DATA_PATH = "/home/frbea320/projects/def-flavielc/frbea320/mscts-analysis/baselines/files"

best_1_0 = 2
best_25_0 = 23
best_25_1 = 17
best_25_2 = 23
best_25_4 = 23
best_25_8 = 21


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

def load_data():
    with open(f"{DATA_PATH}/macro.ijm", "r") as file:
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

    start = tuple(map(lambda a, da: (np.round(a) - da // 2).astype(int), location, bounding))
    end = tuple(map(operator.add, start, bounding))
    if pad:
        padNumpy = []
        for low, up, arrShape in zip(start, end, arrayShape):
            pad_min = -low if low < 0 else 0
            pad_max = up - arrShape if (up - arrShape) > 0 else 0
            padNumpy.append((pad_min, pad_max))
        padNumpy = tuple(padNumpy)

    start = tuple(map(lambda a: np.clip(a, 0, None), start))
    end = tuple(map(lambda a, dmax: np.clip(a, 0, dmax).astype(int), end, array.shape))
    slices = tuple(map(slice, start, end))
    array = array[slices]

    if pad:
        array = np.pad(array, padNumpy, **kwargs)

    return array, slices
    
def get_deltaF(image, center):
    _, slices = get_crop(image, center, (1, 5, 5))
    trace = np.mean(image[:, slices[1], slices[2]], axis=(1, 2))

    # Compute normalized trace
    F0 = baseline(trace, lam=1e6, ratio=1e-10)

    trace_normalized = (trace - F0) / F0
    deltaF_frame = np.round(center[0]).astype(int)
    deltaF = trace_normalized[deltaF_frame]
    return deltaF

def is_empty(prediction):
    num_nonzero = np.count_nonzero(prediction)
    return num_nonzero == 0

def get_centroid_distance(truth_centroids, pred_centroids):
    truth_centroids = np.array(truth_centroids)
    pred_centroids = np.array(pred_centroids)
    distances = cdist(truth_centroids, pred_centroids)
    return np.min(distances)


def get_dice_and_centroid_scores(data, model):
    model_dict = {
        "dice": [],
        "distances": []
    }
    skipped = 0
    for event_type, values in tqdm(data.items(), desc="Event types...", leave=False):
        for movie_id, coords, time_slice, index in tqdm(values, desc="Events..."):
            movie = tifffile.imread(f"{RAW_PATH}/{movie_id}")
            truth_name = glob.glob(f"{EXPERT_PATH}/*-{index}.tif")[0]
            truth = tifffile.imread(truth_name)
            x, y, width, height = coords
            movie_crop = movie[time_slice, y:y+height, x:x+width]
            pred_path = f"{BASE_PATH}/{model}/Quality Control Segmentation/Prediction/{movie_id}".replace(".tif", "_prediction.tif")
            pred = tifffile.imread(pred_path)
            pred = pred[time_slice, y:y+height, x:x+width]
            if is_empty(pred):
                skipped += 1
                continue
            pred_label = skimage.measure.label(pred)
            pred_rprops = skimage.measure.regionprops(pred_label, intensity_image=movie_crop)
            truth_label = skimage.measure.label(truth)
            truth_rprops = skimage.measure.regionprops(truth_label, intensity_image=movie_crop)
            truth_centroids = [r.weighted_centroid for r in truth_rprops]
            pred_centroids = [r.weighted_centroid for r in pred_rprops]
            centroid_distance = get_centroid_distance(truth_centroids, pred_centroids)
            dice = commons.dice(truth, pred)
            model_dict["distances"].append(centroid_distance)
            model_dict["dice"].append(dice)
    print(model_dict)
    print(f"{skipped} empty predictions were found.")
    model_dict["skipped"] = skipped
    return model_dict
            

def main():
    data = load_data()
    data_1_0 = get_dice_and_centroid_scores(data=data, model=UNet_025["UNet3D_complete_1-0"][best_1_0])
    data_25_0 = get_dice_and_centroid_scores(data=data, model=UNet_025["25-0"][best_25_0])
    data_25_1 = get_dice_and_centroid_scores(data=data, model=UNet_025["25-1"][best_25_1])
    data_25_2 = get_dice_and_centroid_scores(data=data, model=UNet_025["25-2"][best_25_2])
    data_25_4 = get_dice_and_centroid_scores(data=data, model=UNet_025["25-4"][best_25_4])
    data_25_8 = get_dice_and_centroid_scores(data=data, model=UNet_025["25-8"][best_25_8])

    models_data = [data_1_0, data_25_0, data_25_1, data_25_2, data_25_4, data_25_8]
    fnames = ["UNet_1-0", "UNet_25-0", "UNet_25-1", "UNet_25-2", "UNet_25-4", "UNet_25-8"]
    for fname, mdata in zip(fnames, models_data):
        with open(f"./dice_vs_centroid_data/{fname}.pkl", "wb") as f:
            pickle.dump(mdata, f)

    

if __name__=="__main__":
    main()