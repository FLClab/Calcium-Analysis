import numpy as np
import matplotlib.pyplot as plt
import glob
from metrics.segmentation import commons
from tqdm import tqdm 
from model_paths import UNet_025
import pandas 
import os
import tifffile
import operator
import scipy.stats
from dice_vs_centroid import load_data
import copy

RAW_PATH = "/home/frbea320/projects/def-flavielc/frbea320/mscts-analysis/data/testset/raw-input"
FLAVIE1_PATH = "/home/frbea320/projects/def-flavielc/frbea320/mscts-analysis/data/testset/segmentation-expert/mask_flavie_1"
FLAVIE2_PATH = "/home/frbea320/projects/def-flavielc/frbea320/mscts-analysis/data/testset/segmentation-expert/mask_flavie_2"
THERESA1_PATH = "/home/frbea320/projects/def-flavielc/frbea320/mscts-analysis/data/testset/segmentation-expert/mask_theresa_1"
THERESA2_PATH = "/home/frbea320/projects/def-flavielc/frbea320/mscts-analysis/data/testset/segmentation-expert/mask_theresa_2"
GROUND_TRUTH_PATH = "/home/frbea320/projects/def-flavielc/frbea320/mscts-analysis/data/testset"


model_path = f"/home/frbea320/projects/def-flavielc/frbea320/mscts-analysis/data/baselines/ZeroCostDL4Mic/{UNet_025['25-4'][23]}"

def normalize_data(data: np.ndarray) -> np.ndarray:
    norm = (data - np.min(data)) / (np.max(data) - np.min(data))
    return norm

def to_rgb(img: np.ndarray) -> np.ndarray:
    img = np.stack((img,)*4, axis=-1)
    img[:, :, -1] = 1
    return img

def invert_graylut(img: np.ndarray) -> np.ndarray:
    return 1 - img

def colorize_pixel(truth: np.ndarray, pred: np.ndarray, i: int, j: int) -> str:
    in_truth = truth[i][j][0] == 1
    in_pred = pred[i][j][0] == 1
    if in_truth and in_pred: 
        color = (0, 1, 0, 1)
    elif in_truth and not in_pred:
        color = (1, 0.4, 0, 1)
    elif not in_truth and in_pred:
        color = (0, 0, 1, 1)
    else:
        color = (0, 0, 0, 0)
    return color

def compute_dice_example(data) -> None:
    idx = 0
    for _, values in tqdm(data.items(), desc="Event types..."):
        for movie_id, coords, time_slice, index in tqdm(values, desc="Events..."):
            movie = tifffile.imread(f"{RAW_PATH}/{movie_id}")
            truth_name = glob.glob(f"{THERESA1_PATH}/*-{index}.tif")[0]
            minifinder_name = os.path.join(GROUND_TRUTH_PATH, "minifinder", movie_id.replace(".tif", ".npy"))
            minifinder_prediction = np.load(minifinder_name)
            truth = tifffile.imread(truth_name)
            x, y, width, height = coords
            movie_slice = movie[time_slice, y:y+height, x:x+width]
            minifinder_pred = minifinder_prediction[time_slice, y:y+height, x:x+width]
            path = f"{model_path}/Quality Control Segmentation/Prediction/{movie_id}".replace(".tif", "_prediction.tif")
            pred = tifffile.imread(path)
            pred = pred[time_slice, y:y+height, x:x+width]
            dice = commons.dice(truth, pred)
            mf_dice = commons.dice(truth, minifinder_pred)
            create_overlay(
                raw=movie_slice,
                truth=truth,
                mf_pred=minifinder_pred,
                prediction=pred,
                mf_dice=mf_dice,
                dice=dice,
                save_idx=idx,
            )
            idx += 1

def save_overlay(original: np.ndarray, raw: np.ndarray, raw_copy: np.ndarray, truth: np.ndarray, mf_dice: float, dice: float, save_idx: int) -> None:
    fig, axs = plt.subplots(1, 4, figsize=(10,10))
    axs[0].imshow(raw)
    axs[1].imshow(raw_copy)
    axs[2].imshow(truth, cmap='gray')
    axs[3].imshow(original)
    for ax in axs:
        ax.set_xticks([])
        ax.set_yticks([])
    axs[0].set_title(f"MF dice\n{round(mf_dice, 4)}")
    axs[1].set_title(f"Model dice\n{round(dice, 4)}")
    # fig.savefig(f"./UNet_seg_examples/example_{save_idx}.png")
    fig.savefig(f"./UNet_seg_examples/example_{save_idx}.pdf", transparent=True, bbox_inches='tight')


def create_overlay(raw: np.ndarray, truth: np.ndarray, mf_pred: np.ndarray, prediction: np.ndarray, mf_dice: float, dice: float, save_idx: int) -> None:
    assert raw.shape == truth.shape
    assert truth.shape == prediction.shape
    min_ax = min(raw.shape)
    raw = raw[:min_ax, :min_ax]
    truth = truth[:min_ax, :min_ax]
    mf_pred = mf_pred[:min_ax, :min_ax]
    prediction = prediction[:min_ax, :min_ax]
    assert raw.shape[0] == raw.shape[1]
    raw = to_rgb(invert_graylut(normalize_data(raw)))
    truth = to_rgb(normalize_data(truth))
    mf_pred = to_rgb(normalize_data(mf_pred))
    prediction = to_rgb(normalize_data(prediction))
    untouched_raw = copy.deepcopy(raw)
    raw_copy = copy.deepcopy(raw)
    for i in range(raw.shape[0]):
        for j in range(raw.shape[1]):
            pred_color = colorize_pixel(truth, prediction, i, j)
            mf_color = colorize_pixel(truth, mf_pred, i, j)
            if mf_color != (0, 0, 0, 0):
                raw[i, j] = mf_color
            if pred_color != (0,0,0,0):
                raw_copy[i, j] = pred_color
    save_overlay(
        original=untouched_raw,
        raw=raw,
        raw_copy=raw_copy,
        truth=truth,
        mf_dice=mf_dice,
        dice=dice,
        save_idx=save_idx
    )


def main():
    data = load_data()
    compute_dice_example(data=data)

if __name__=="__main__":
    main()
