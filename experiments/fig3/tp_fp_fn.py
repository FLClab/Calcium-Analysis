import numpy as np
import matplotlib.pyplot as plt
import pickle
import tifffile
from model_paths import UNet_025
from tqdm import tqdm
import os
import glob
import pandas
import scipy.sparse.linalg
from scipy.spatial import distance
from scipy.optimize import linear_sum_assignment
# from metrics import CentroidDetectionError
import operator
import argparse 

parser = argparse.ArgumentParser()
parser.add_argument("--models", type=str, default="UNet-4-0")
args = parser.parse_args()



RESULTS_PATH = "/home/frbea320/projects/def-flavielc/anbil106/MSCTS-Analysis/testset/UNet3D/"
GROUND_TRUTH_PATH = "/home/frbea320/projects/def-flavielc/frbea320/mscts-analysis/data/testset"
RAW_PATH = "/home/frbea320/projects/def-flavielc/frbea320/mscts-analysis/data/testset/raw-input"
BASE_PATH = f"/home/frbea320/projects/def-flavielc/frbea320/mscts-analysis/data/baselines/ZeroCostDL4Mic"

best_1_0 = 2
best_25_0 = 23
best_25_1 = 23 
best_25_2 = 23
best_25_4 = 23
best_25_8 = 6
best_25_16 = 5
best_25_32 = 6
best_25_64 = 9
best_25_128 = 3
best_25_256 = 18

def load_models(model_seeds: str = args.models):
    if args.models == "UNet-4-0":
        models = UNet_025["UNet3D_complete_1-0"]
        return models
    elif args.models == "UNet-64-0":
        models = UNet_025["25-64"]
        return models
    elif models == "best":
        models = [
            UNet_025["UNet3D_complete_1-0"][best_1_0],
            UNet_025["25-0"][best_25_0],
            UNet_025["25-1"][best_25_1],
            UNet_025["25-2"][best_25_2],
            UNet_025["25-4"][best_25_4],
            UNet_025["25-8"][best_25_8],
            UNet_025["25-16"][best_25_16],
            UNet_025["25-32"][best_25_32],
            UNet_025["25-64"][best_25_64],
            UNet_025["25-128"][best_25_128],
            UNet_025["25-256"][best_25_256]
        ]
        return models
    else:
        exit("The models requested are not supported yet")


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


def filter_rprops(tlen: list, xlen: list, ylen: list,  constraints: dict):
    """
    Enleve les événements selon trop petit selon différents critères
    """

    bool_array = []
    for t, y, x in zip(tlen, ylen, xlen):


        # Constraints check
        good = True
        if 'minimal_time' in constraints.keys():
            if t < constraints['minimal_time']:
                good = False
        if 'minimal_height' in constraints.keys():
            if y < constraints['minimal_height']:
                good = False
        if 'minimal_width' in constraints.keys():
            if x < constraints['minimal_width']:
                good = False
        bool_array.append(good)
    return bool_array


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
    epsilon = 1e-3 if tp + fp == 0 else 0
    return tp / (tp + fp + epsilon)

def get_recall(tp, fn):
    epsilon = 1e-3 if tp + fn == 0 else 0
    return tp / (tp+fn + epsilon)

def get_f1(recall, precision):
    epsilon = 1e-3 if precision + recall == 0 else 0
    return (2*precision * recall) / (precision+recall+epsilon)

def compute_histogram(deltaF_tp, deltaF_fp, deltaF_fn, model):
    max_tp = int(np.ceil(max(deltaF_tp)))
    max_fp = int(np.ceil(max(deltaF_fp)))
    max_fn = int(np.ceil(max(deltaF_fn))) 
    max_x = max([max_tp, max_fp, max_fn])
    bins_for_all = np.arange(0, 7, 0.5)
    tp_hist, tp_edges = np.histogram(deltaF_tp, bins_for_all)
    fp_hist, fp_edges = np.histogram(deltaF_fp, bins_for_all)
    fn_hist, fn_edges = np.histogram(deltaF_fn, bins_for_all)
    print(f"\n\n {model} # detected events: {np.sum(tp_hist) + np.sum(fp_hist) + np.sum(fn_hist)}\n\n")
    # fig = plt.figure()
    # plt.bar(x=tp_edges[:-1], height=tp_hist, width=0.45, color='tab:green', label='TP', align='edge')
    # plt.bar(x=fp_edges[:-1], height=fp_hist, width=0.45, bottom=tp_hist, color='tab:blue', label='FP', align='edge')
    # plt.bar(x=fn_edges[:-1], height=fn_hist, width=0.45, bottom=[tp + fp for tp, fp in zip(tp_hist, fp_hist)], color='tab:orange', label='FN', align='edge')
    # plt.ylabel("# mSCTs")
    # plt.xlabel('Delta F / F')
    # plt.legend()
    # fig.savefig(f"./figures/paper_figures/{model}_detection_bars.png")
    # fig.savefig(f"./figures/paper_figures/{model}_detection_bars.pdf", transparent=True, bbox_inches='tight')
    # plt.close(fig)
    return (tp_hist, tp_edges), (fp_hist, fp_edges), (fn_hist, fn_edges)

def evaluate_against_expert(model):
    all_fp, all_tp , all_fn = 0, 0, 0
    deltaF_tp, deltaF_fp, deltaF_fn = [], [], []
    for f in tqdm(manual_expert_files, desc="Manual annotations..."):
        fname = f.split("/")[-1].split("_")[1].split(".")[0]
        print(f"*********** Video {f} ****************")
        movie = get_movie(fname)
        data = pandas.read_csv(f)
        truth_centroids = np.stack((data["Slice"], data["Y"], data["X"]), axis=-1)
        pred_path = f"{BASE_PATH}/{model}/Quality Control Segmentation/QC_regionprops_{model}.pkl"
        try:
            pred_dict = load_dict(pred_path)
            tmin = pred_dict[f'{fname}_prediction.tif']['bbox-0']
            ymin = pred_dict[f'{fname}_prediction.tif']['bbox-1']
            xmin = pred_dict[f'{fname}_prediction.tif']['bbox-2']
            tmax = pred_dict[f'{fname}_prediction.tif']['bbox-3']
            ymax = pred_dict[f'{fname}_prediction.tif']['bbox-4']
            xmax = pred_dict[f'{fname}_prediction.tif']['bbox-5']
            tlen = tmax - tmin
            xlen = xmax - xmin
            ylen = ymax - ymin
            bool_array = filter_rprops(tlen=tlen, ylen=ylen, xlen=xlen, constraints=POSTPROCESS_PARAMS)
            pred_centroids = np.stack((
                pred_dict[f'{fname}_prediction.tif']['centroid-0'],
                pred_dict[f'{fname}_prediction.tif']['centroid-1'],
                pred_dict[f'{fname}_prediction.tif']['centroid-2']
            ), axis=-1)
            pred_centroids = pred_centroids[bool_array]
        except FileNotFoundError:
            missing_data.append(pred_path)
            print(f"Model {model} does not predict movie {fname}")
            continue 
        except KeyError:
            print(f"Model {model} does not predict movie {fname}")
            continue

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

def plot_fancy_histogram(data, expr):
    print(data.shape)
    # xticks = [0, 2, 4, 6, 8]
    # xlabels = ['0', '1', '2', '3', '4', '5', '6']
    # yticks = [0, 1, 2, 3, 4]
    # ylabels = ['4-0', '1-0', '1-1', '1-2', '1-4', '1-8']
    # xticks = [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
    fig = plt.figure()
    plt.imshow(data, cmap='RdPu')
    plt.xlabel('Delta F / F')
    plt.ylabel('Models')
    # plt.xticks(ticks=xticks, labels=xlabels)
    # plt.yticks(ticks=yticks, labels=ylabels)
    plt.colorbar(orientation="vertical") 
    fig.savefig(f"./figures/paper_figures/{expr}_colorcoded_histogram.png", bbox_inches='tight')
    fig.savefig(f"./figures/paper_figures/{expr}_colorcoded_histogram.pdf", transparent=True, bbox_inches='tight')
    plt.close(fig)

def main():
    models = load_models()
    tp_array = np.zeros((len(models), 13))
    fp_array = np.zeros((len(models), 13))
    fn_array = np.zeros((len(models), 13))
    for i, model in enumerate(tqdm(models, desc="Models...")):
        tp, fp, fn = evaluate_against_expert(model)
        tp_array[i] = tp[0]
        fp_array[i] = fp[0]
        fn_array[i] = fn[0]
    
    # tp_array = tp_array[:, :6]
    # fp_array = fp_array[:, :6]
    # fn_array = fn_array[:, :6]
    np.savez(f"./{args.models}__tp_fp_fn_data", tp_array=tp_array, fp_array=fp_array, fn_array=fn_array)
    # tp_array = tp_array/tp_array.sum(axis=0)[None, :]
    # fp_array = fp_array/fp_array.sum(axis=0)[None, :]
    # fn_array = fn_array/fn_array.sum(axis=0)[None, :] 
    # plot_fancy_histogram(tp_array, "TP")
    # plot_fancy_histogram(fp_array, "FP")
    # plot_fancy_histogram(fn_array, "FN")
    # print("\n\n")
    # print(f"There were {len(missing_data)} missing files. Here they are:")
    for data in missing_data:
        print(data)
    print("*************** DONE *****************")

if __name__=="__main__":
    main()
    