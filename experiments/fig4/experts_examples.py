import numpy as np
import matplotlib.pyplot as plt
import tifffile
from tqdm import tqdm 
import glob 
import argparse 
from metrics.segmentation import commons 
from dice_vs_centroid import load_data 
from typing import List
from segmentation_examples import normalize_data, to_rgb, invert_graylut, colorize_pixel
import copy
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--expert", type=str, default="expert1")
args = parser.parse_args()

FLAVIE1_PATH = "/home/frbea320/projects/def-flavielc/frbea320/mscts-analysis/data/testset/segmentation-expert/mask_flavie_1"
FLAVIE2_PATH = "/home/frbea320/projects/def-flavielc/frbea320/mscts-analysis/data/testset/segmentation-expert/mask_flavie_2"
THERESA1_PATH = "/home/frbea320/projects/def-flavielc/frbea320/mscts-analysis/data/testset/segmentation-expert/mask_theresa_1"
THERESA2_PATH = "/home/frbea320/projects/def-flavielc/frbea320/mscts-analysis/data/testset/segmentation-expert/mask_theresa_2"
RAW_PATH = "/home/frbea320/projects/def-flavielc/frbea320/mscts-analysis/data/testset/raw-input"
GROUND_TRUTH_PATH = "/home/frbea320/projects/def-flavielc/frbea320/mscts-analysis/data/testset"



def inter_expert(data:dict) -> None:
    idx = 0
    min_dice = 1.0
    max_dice = 0.0
    for _, values in tqdm(data.items(), desc="Events types..."):
        for movie_id, coords, time_slice, index in tqdm(values, desc="Events..."):
            movie = tifffile.imread(f"{RAW_PATH}/{movie_id}")
            expert1_name = glob.glob(f"{FLAVIE2_PATH}/*-{index}.tif")[0]
            expert2_name = glob.glob(f"{THERESA2_PATH}/*-{index}.tif")[0]
            expert1 = tifffile.imread(expert1_name)
            expert2 = tifffile.imread(expert2_name)
            x, y, width, height = coords
            movie_slice = movie[time_slice, y:y+height, x:x+width]
            dice = commons.dice(expert1, expert2)
            if dice < min_dice:
                min_dice = dice
                create_overlay(
                    raw=movie_slice,
                    expert1=expert1,
                    expert2=expert2,
                    dice=dice,
                    extreme='minimum'
                )
                idx+=1 
            elif dice > max_dice:
                max_dice = dice
                create_overlay(
                    raw=movie_slice,
                    expert1=expert1,
                    expert2=expert2,
                    dice=dice,
                    extreme='maximum'
                )
                idx+=1 
            else:
                continue


def intra_expert(data: dict, session1: str, session2: str) -> None:
    idx = 0
    min_dice = 1.0
    max_dice = 0.0
    for _, values in tqdm(data.items(), desc="Event types..."):
        for movie_id, coords, time_slice, index in tqdm(values, desc="Events..."):
            movie = tifffile.imread(f"{RAW_PATH}/{movie_id}")
            session1_name = glob.glob(f"{session1}/*-{index}.tif")[0]
            session2_name = glob.glob(f"{session2}/*-{index}.tif")[0]
            session1_pred = tifffile.imread(session1_name)
            session2_pred = tifffile.imread(session2_name)
            x, y, width, height = coords
            movie_slice = movie[time_slice, y:y+height, x:x+width]
            dice = commons.dice(session1_pred, session2_pred)
            if dice < min_dice:
                min_dice = dice
                create_intra_overlay(
                    raw=movie_slice,
                    session1=session1_pred,
                    session2=session2_pred,
                    dice=dice,
                    extreme='minimum'
                )
                idx += 1
            elif dice > max_dice:
                max_dice = dice
                create_intra_overlay(
                    raw=movie_slice,
                    session1=session1_pred,
                    session2=session2_pred,
                    dice=dice,
                    extreme='maximum'
                )
                idx += 1
            else:
                continue

def create_overlay(raw: np.ndarray, expert1: np.ndarray, expert2: np.ndarray, dice: float, extreme: str) -> None:
    assert raw.shape == expert1.shape
    assert expert1.shape == expert2.shape
    min_ax = min(raw.shape)
    raw = raw[:min_ax, :min_ax]
    expert1 = expert1[:min_ax, :min_ax]
    expert2 = expert2[:min_ax, :min_ax]
    assert raw.shape[0] == raw.shape[1]
    raw = to_rgb(invert_graylut(normalize_data(raw)))
    expert1 = to_rgb(normalize_data(expert1))
    expert2 = to_rgb(normalize_data(expert2))
    original = copy.deepcopy(raw)
    for i in range(raw.shape[0]):
        for j in range(raw.shape[1]):
            color = colorize_pixel(expert1, expert2, i, j)
            if color != (0, 0, 0, 0):
                raw[i, j] = color
    save_overlay(
        original=original,
        overlay=raw,
        expert1=expert1,
        expert2=expert2,
        dice=dice, 
        extreme=extreme
    )

def create_intra_overlay(raw: np.ndarray, session1: np.ndarray, session2: np.ndarray, dice: float, extreme: str) -> None:
    assert raw.shape == session1.shape
    assert session1.shape == session2.shape
    min_ax = min(raw.shape)
    raw = raw[:min_ax, :min_ax]
    session1 = session1[:min_ax, :min_ax]
    session2 = session2[:min_ax, :min_ax]
    assert raw.shape[0] == raw.shape[1]
    raw = to_rgb(invert_graylut(normalize_data(raw)))
    session1 = to_rgb(normalize_data(session1))
    session2 = to_rgb(normalize_data(session2))
    original = copy.deepcopy(raw)
    for i in range(raw.shape[0]):
        for j in range(raw.shape[1]):
            color = colorize_pixel(session1, session2, i , j)
            if color != (0, 0, 0, 0):
                raw[i, j] = color
    save_intra_overlay(
        original=original,
        overlay=raw, 
        session1=session1,
        session2=session2, 
        dice=dice,
        extreme=extreme
    )

def save_intra_overlay(original: np.ndarray, overlay: np.ndarray, session1: np.ndarray, session2: np.ndarray, dice: float, extreme: str) -> None:
    fig, axs = plt.subplots(1, 4, figsize=(10,10))
    axs[0].imshow(original)
    axs[1].imshow(session1)
    axs[2].imshow(session2)
    axs[3].imshow(overlay)
    for ax, title in zip(axs, ["Original", f"Session 1\n{dice}", "Session 2", "Overlay"]):
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title)
    fig.savefig(f"./expert_seg_examples/intra_{args.expert}/example_{extreme}.pdf", transparent=True, bbox_inches='tight')

def save_overlay(original: np.ndarray, overlay: np.ndarray, expert1: np.ndarray, expert2: np.ndarray, dice: float, extreme: str):
    fig, axs = plt.subplots(1, 4, figsize=(10, 10))
    axs[0].imshow(original)
    axs[1].imshow(expert1)
    axs[2].imshow(expert2)
    axs[3].imshow(overlay)
    for ax, title in zip(axs, ['Original', f'Expert 1\n{dice}', 'Expert 2', 'Overlay']):
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title)
    fig.savefig(f"./expert_seg_examples/inter/example_{extreme}_2.pdf", transparent=True, bbox_inches='tight')    

def main():
    data = load_data()
    session1 = FLAVIE1_PATH if args.expert == "expert1" else THERESA1_PATH
    session2 = FLAVIE2_PATH if args.expert == "expert1" else THERESA2_PATH
    intra_expert(data=data, session1=session1, session2=session2)


if __name__=="__main__":
    main()