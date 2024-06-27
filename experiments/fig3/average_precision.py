import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import pickle
import utils
from model_paths import UNet_025, UNET_4, STARDIST_025
from scipy.interpolate import interp1d
from tqdm import tqdm
import argparse
import colorsys
from matplotlib.colors import LinearSegmentedColormap
import matplotlib
from sklearn.metrics import auc

parser = argparse.ArgumentParser()
parser.add_argument("--backbone", type=str, required=True)
args = parser.parse_args()

def scale_lightness(rgb, scale_l):
    # convert rgb to hls
    h, l, s = colorsys.rgb_to_hls(*rgb)
    # manipulate h, l, s values and return as rgb
    return colorsys.hls_to_rgb(h, min(1, l * scale_l), s = s)


cmap_og = matplotlib.colors.LinearSegmentedColormap.from_list(
    name="nice-prism",
    colors=["#5F4690","#1D6996","#38A6A5","#0F8554","#73AF48","#EDAD08","#E17C05","#CC503E","#94346E"]
)
matplotlib.colormaps.register(cmap=cmap_og, force=True)
matplotlib.colormaps.register(cmap=cmap_og.reversed(), force=True)



RESULTS_PATH = "/home/frbea320/projects/def-flavielc/anbil106/MSCTS-Analysis/testset/UNet3D/"
SD_OLD_PATH = "/home/frbea320/scratch/StarDist3D"
SD_NEW_PATH = "/home/frbea320/projects/def-flavielc/frbea320/mscts-analysis/data/testset/StarDist3D"

OMNIRECALL = np.linspace(0, 1, 100)

permission_denied = [

]

file_not_found = [

]

def load_dict(path):
    data_dict = pickle.load(open(f"{path}/results.pkl", "rb"))
    return data_dict

def compute_average_precision(recall: np.ndarray, precision: np.ndarray):
    n = recall.shape[0]
    ap = 0
    for i in range(1, n):
        r_curr = recall[i]
        r_prev = recall[i - 1]
        p_curr = precision[i]
        temp = (r_curr - r_prev)*p_curr
        ap += temp
    return ap

def clip_when_decreasing(recall):
    max_index = recall.index(max(recall))
    return max_index


def aggregate_unet_seeds(models):
    recall = np.zeros((len(models), 100))
    precision = np.zeros((len(models), 100))
    best_f1 = 0
    for i, seed in enumerate(models):
        path = os.path.join(RESULTS_PATH, seed)
        data = load_dict(path)
        # recall[i] = data["recall"]
        # precision[i] = data["precision"]
        recall_lst = data["recall"]
        precision_lst = data["precision"]
        
        recall_rev = recall_lst[::-1]
        precision_rev = precision_lst[::-1]
        max_index = clip_when_decreasing(recall_rev)
        recall_rev, precision_rev = recall_rev[:max_index], precision_rev[:max_index]
        f = interp1d(recall_rev, precision_rev, assume_sorted=True, bounds_error=False, fill_value=(precision_rev[0], precision_rev[-1]))
        y_rev = f(OMNIRECALL)
        recall[i] = OMNIRECALL
        precision[i] = y_rev
    return recall, precision


def aggregate_stardist_seeds(models, pu_config):
    recall = np.zeros((len(models), 100))
    precision = np.zeros((len(models), 100))
    base_path = SD_OLD_PATH if pu_config == "4-0" else SD_NEW_PATH
    for i, seed in enumerate(models):
        path = os.path.join(base_path, seed)
        try:
            data = load_dict(path)
        except PermissionError:
            permission_denied.append(path)
            continue
        except FileNotFoundError:
            file_not_found.append(path)
            continue
        recall_lst = data["recall"].tolist()
        precision_lst = data["precision"].tolist()
        recall_rev = recall_lst[::-1]
        precision_rev = precision_lst[::-1]

        max_index = clip_when_decreasing(recall_rev)
        recall_rev, precision_rev = recall_rev[:max_index], precision_rev[:max_index]
        f = interp1d(recall_rev, precision_rev, assume_sorted=True, bounds_error=False, fill_value=(precision_rev[0], precision_rev[-1]))
        y_rev = f(OMNIRECALL)
        recall[i] = OMNIRECALL
        precision[i] = y_rev
    return recall, precision

def main_unet():
    for i, model_config in enumerate(tqdm(UNet_025.keys(), desc="UNet models...")):
        recall, precision = aggregate_unet_seeds(UNet_025[model_config])
        recall = np.mean(recall, axis=0)
        precision = np.mean(precision, axis=0)
        ap = compute_average_precision(recall=recall, precision=precision)
        print(f"{model_config} AP = {ap}")


def main_stardist():
    for i, model_config in enumerate(tqdm(STARDIST_025, desc="StarDist models...")):
        recall, precision = aggregate_stardist_seeds(STARDIST_025[model_config], pu_config=model_config)
        recall = np.mean(recall, axis=0)
        precision = np.mean(precision, axis=0)
        ap = compute_average_precision(recall=recall, precision=precision)
        print(f"{model_config} AP = {ap}")



def main():
    if args.backbone == "UNet":
        main_unet()

    else:
        main_stardist()

if __name__=="__main__":
    main()