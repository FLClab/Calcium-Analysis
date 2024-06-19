import numpy as np
import matplotlib.pyplot as plt
import argparse
import pickle 
from model_paths import STARDIST_025
import os
from sklearn.metrics import auc

SD_OLD_PATH = "/home/frbea320/scratch/StarDist3D"
SD_NEW_PATH = "/home/frbea320/projects/def-flavielc/frbea320/mscts-analysis/data/testset/StarDist3D"

def load_dict(path: str) -> dict:
    data = pickle.load(open(f"{path}/results.pkl", "rb"))
    return data

def clip_when_decreasing(recall):
    max_index = recall.index(max(recall))
    return max_index

def get_AUPR(pu_config, models):
    for i, seed in enumerate(models):
        path = os.path.join(SD_OLD_PATH, seed) if pu_config == "4-0" else os.path.join(SD_NEW_PATH, seed)
        try:
            data = load_dict(path=path)
        except FileNotFoundError:
            continue
        recall_lst = list(data['recall'])
        precision_lst = list(data['precision'])
        recall_rev = recall_lst[::-1]
        precision_rev = precision_lst[::-1]
        max_index = clip_when_decreasing(recall_rev)
        recall_rev, precision_rev = recall_rev[:max_index], precision_rev[:max_index]
        score = auc(x=recall_rev, y=precision_rev)
        print(f"{pu_config}, seed {seed} --> {score}")
    exit()

def main():
    for key in STARDIST_025.keys():
        get_AUPR(pu_config=key, models=STARDIST_025[key])
        print("\n")

if __name__=="__main__":
    main()