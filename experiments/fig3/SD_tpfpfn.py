import numpy as np
import matplotlib.pyplot as plt
import pickle
import tifffile
from model_paths import STARDIST_025
from tqdm import tqdm
import os
import pandas
from typing import Tuple
import glob
import operator
import scipy.sparse.linalg
from scipy.spatial import distance
from scipy.optimize import linear_sum_assignment
import skimage.measure
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--models", type=str, default="StarDist-4-0")
args = parser.parse_args()

SD_OLD_PATH = "/home/frbea320/scratch/StarDist3D"
SD_NEW_PATH = "/home/frbea320/projects/def-flavielc/frbea320/mscts-analysis/data/testset/StarDist3D"
GROUND_TRUTH_PATH = "/home/frbea320/projects/def-flavielc/frbea320/mscts-analysis/data/testset"
RAW_PATH = "/home/frbea320/projects/def-flavielc/frbea320/mscts-analysis/data/testset/raw-input"



best_4_0 = 4
best_1_0 = 9
best_1_1 = 2
best_1_2 = 23
best_1_4 = 16
best_1_8 = 23
best_1_16 = 8
best_1_32 = 9
best_1_64 = 6
best_1_128 = 19
best_1_256 = 5


best_models = [
    f"/home/frbea320/scratch/StarDist3D/{STARDIST_025['4-0'][best_4_0]}",
    f"/home/frbea320/projects/def-flavielc/frbea320/mscts-analysis/data/testset/StarDist3D/{STARDIST_025['1-0'][best_1_0]}",
    f"/home/frbea320/projects/def-flavielc/frbea320/mscts-analysis/data/testset/StarDist3D/{STARDIST_025['1-1'][best_1_1]}",
    f"/home/frbea320/projects/def-flavielc/frbea320/mscts-analysis/data/testset/StarDist3D/{STARDIST_025['1-2'][best_1_2]}",
    f"/home/frbea320/projects/def-flavielc/frbea320/mscts-analysis/data/testset/StarDist3D/{STARDIST_025['1-4'][best_1_4]}",
    f"/home/frbea320/projects/def-flavielc/frbea320/mscts-analysis/data/testset/StarDist3D/{STARDIST_025['1-8'][best_1_8]}",
    f"/home/frbea320/projects/def-flavielc/frbea320/mscts-analysis/data/testset/StarDist3D/{STARDIST_025['1-16'][best_1_16]}",
    f"/home/frbea320/projects/def-flavielc/frbea320/mscts-analysis/data/testset/StarDist3D/{STARDIST_025['1-32'][best_1_32]}",
    f"/home/frbea320/projects/def-flavielc/frbea320/mscts-analysis/data/testset/StarDist3D/{STARDIST_025['1-64'][best_1_64]}",
    f"/home/frbea320/projects/def-flavielc/frbea320/mscts-analysis/data/testset/StarDist3D/{STARDIST_025['1-128'][best_1_128]}",
    f"/home/frbea320/projects/def-flavielc/frbea320/mscts-analysis/data/testset/StarDist3D/{STARDIST_025['1-256'][best_1_256]}"
]

def load_models(model_seeds: str = args.models):
    if args.models == "StarDist-4-0":
        root = "/home/frbea320/scratch/StarDist3D"
        models = [
            f"{root}/{item}" for item in STARDIST_025['4-0']
        ]
        return models
    elif args.models == "StarDist-1-1":
        root = "/home/frbea320/projects/def-flavielc/frbea320/mscts-analysis/data/testset/StarDist3D"
        models = [
            f"{root}/{item}" for item in STARDIST_025['1-1']
        ]
        return models
    elif args.models == 'best':
        models = best_models
        return models
    else:
        exit("The models requested are not supported yet.")
        

POSTPROCESS_PARAMS = {
            'minimal_time': 3,
            'minimal_height': 3,
            'minimal_width': 3,
        }

manual_expert_files = glob.glob(os.path.join(GROUND_TRUTH_PATH, "manual-expert", "*.csv"))

missing_data = []

def load_dict(path):
    data_dict = pickle.load(open(path, "rb"))
    return data_dict

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
    

def get_precision(tp, fp):
    epsilon = 1e-5 if tp + fp == 0 else 0
    return tp / (tp + fp + epsilon)

def get_recall(tp, fn):
    epsilon = 1e-5 if tp + fn == 0 else 0
    return tp / (tp+fn + epsilon)

def get_f1(recall, precision):
    epsilon = 1e-5 if precision + recall == 0 else 0
    return (2*precision * recall) / (precision+recall+epsilon)

def evaluate_against_expert(model: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    all_tp, all_fp, all_fn = 0, 0, 0
    deltaF_tp, deltaF_fp, deltaF_fn = [], [], []
    for f in tqdm(manual_expert_files, desc="Manual annotations..."):
        fname = f.split("/")[-1].split("_")[1].split(".")[0]
        movie = get_movie(fname)
        data = pandas.read_csv(f)
        truth_centroids = np.stack((data["Slice"], data["Y"], data["X"]), axis=-1)
        
        pred_path = f"{model}/{fname}.npz"
        try:
            pred = np.load(pred_path)['label']
        except FileNotFoundError:
            missing_data.append(pred_path)
            print(f"Model {model} does not predict movie {fname}")
            continue
        except KeyError:
            print(f"Model {model} does not predict movie {fname}")
            continue
        pred_label = skimage.measure.label(pred)
        pred_rprops = skimage.measure.regionprops(label_image=pred_label, intensity_image=movie)
        pred_rprops = filter_rprops(regions=pred_rprops, constraints=POSTPROCESS_PARAMS)
        pred_centroids = [r.weighted_centroid for r in pred_rprops]
        truth_deltaF = [get_deltaF(movie, c) for c in truth_centroids]
        pred_deltaF = [get_deltaF(movie, c ) for c in pred_centroids]
        cost_matrix = compute_cost_matrix(truth_centroids, pred_centroids)
        truth_couple, pred_couple = assign_hungarian(cost_matrix)
        fn = get_fn(cost_matrix, truth_couple)
        fp = get_fp(cost_matrix, pred_couple)
        print(f"TP = {pred_couple.shape}")
        print(f"FP = {fp.shape}")
        print(f"FN = {fn.shape}")
        all_tp += pred_couple.shape[0]
        all_fp += fp.shape[0]
        all_fn += fn.shape[0]
        ##### 
        F_tp = [pred_deltaF[i] for i in pred_couple.tolist()]
        F_fp = [pred_deltaF[i] for i in fp.tolist()]
        F_fn = [truth_deltaF[i] for i in fn.tolist()]
        deltaF_tp += F_tp
        deltaF_fp += F_fp
        deltaF_fn += F_fn
        ######
    precision = get_precision(all_tp, all_fp)
    recall = get_recall(all_tp, all_fn)
    f1 = get_f1(recall, precision)
    print(f"\n*** {model} ***")
    print(f"Precision = {precision}")
    print(f"Recall = {recall}")
    print(f"F1-score = {f1}")
    print("******\n")
    tp_data, fp_data, fn_data = compute_histogram(deltaF_tp, deltaF_fp, deltaF_fn, model)
    return tp_data, fp_data, fn_data

def compute_histogram(deltaF_tp, deltaF_fp, deltaF_fn, model):
    max_tp = int(np.ceil(max(deltaF_tp)))
    max_fp = int(np.ceil(max(deltaF_fp)))
    max_fn = int(np.ceil(max(deltaF_fn))) 
    max_x = max([max_tp, max_fp, max_fn])
    bins_for_all = np.arange(0, 7, 0.5)
    tp_hist, tp_edges = np.histogram(deltaF_tp, bins_for_all)
    fp_hist, fp_edges = np.histogram(deltaF_fp, bins_for_all)
    fn_hist, fn_edges = np.histogram(deltaF_fn, bins_for_all)
    return tp_hist, fp_hist, fn_hist

def main():
    models = load_models()
    tp_array = np.zeros((len(models), 13))
    fp_array = np.zeros((len(models), 13))
    fn_array = np.zeros((len(models), 13))
    for i, model in enumerate(tqdm(models, desc="Models...")):
        tp, fp, fn = evaluate_against_expert(model)
        tp_array[i] = tp
        fp_array[i] = fp
        fn_array[i] = fn
    # tp_array = tp_array[:, :6]
    # fp_array = fp_array[:, :6]
    # fn_array = fn_array[:, :6]
    np.savez(f"./{args.models}__tp_fp_fn_data", tp_array=tp_array, fp_array=fp_array, fn_array=fn_array)
    print(f"There were {len(missing_data)} missing files. Here they are:")
    for data in missing_data:
        print(data)
    print("*************** DONE *****************")

if __name__=="__main__":
    main()
