import numpy as np
import matplotlib.pyplot as plt
import argparse 
import pickle
from model_paths import UNet_025, STARDIST_025
from scipy.interpolate import interp1d
import os
from sklearn.metrics import auc
from operator import itemgetter
from scipy.interpolate import interp1d


best_seeds = {
    "UNet3D_complete_1-0": 2, # 1 or 2
    "25-0": 23,
    "25-1": 20, # 17 or 20 or 23
    "25-2": 23,
    "25-4": 23,
    "25-8": 6,
    "25-16": 5,
    "25-32": 5,
    "25-64": 9,
    "25-128": 3,
    "25-256": 18
}


RESULTS_PATH = "/home/frbea320/projects/def-flavielc/anbil106/MSCTS-Analysis/testset/UNet3D/"

SD_OLD_PATH = "/home/frbea320/scratch/StarDist3D"
SD_NEW_PATH = "/home/frbea320/projects/def-flavielc/frbea320/mscts-analysis/data/testset/StarDist3D"

OMNIRECALL = np.linspace(0, 1, 100)

def load_dict(path):
    data_dict = pickle.load(open(f"{path}/results.pkl", "rb"))
    return data_dict

def clip_when_decreasing(recall):
    max_index = recall.index(max(recall))
    return max_index

def get_AUPR(pu_config, models):
    aucs = []
    for i, seed in enumerate(models):
        # base_path = SD_OLD_PATH if pu_config == "4-0" else SD_NEW_PATH # for stardist
        path = os.path.join(RESULTS_PATH, seed)
        try:
            data = load_dict(path)
        except FileNotFoundError:
            print(f"{pu_config}, seed does not have results.pkl file")
            continue
    
        recall_lst = data["recall"]#.tolist() uncomment with StarDist
        precision_lst = data["precision"]#.tolist()
        recall_rev = recall_lst[::-1]
        precision_rev = precision_lst[::-1]
        max_index = clip_when_decreasing(recall_rev)
        recall_rev, precision_rev = recall_rev[:max_index], precision_rev[:max_index]
        f = interp1d(recall_rev, precision_rev, assume_sorted=True, bounds_error=False, fill_value=(precision_rev[0], precision_rev[-1]))
        y_rev = f(OMNIRECALL)
        temp_p = y_rev[::-1]
        temp_r = OMNIRECALL[::-1]
        score = auc(x=temp_r[27:-27], y=temp_p[27:-27])
        aucs.append(score)
        print(f"{pu_config}, seed {seed} --> {score}")
    return aucs

def plot_seeds(pu_config, models):
    # NUM_COLORS = 25
    # cm = plt.get_cmap('copper_r')
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # ax.set_prop_cycle(color=[cm(1.*i/NUM_COLORS) for i in range(NUM_COLORS)])
    x = OMNIRECALL[::-1]
    for i, seed in enumerate(models):
        seed_id = seed[-9:]
        # if not seed_id.startswith("1") and not seed_id.startswith("0"):
        #     continue
        path = os.path.join(RESULTS_PATH, seed)
        data = load_dict(path)
        recall_lst = data["recall"]
        precision_lst = data["precision"]
        fp_per_video = data["false_positive_per_video"][32]
    
        recall_rev = recall_lst[::-1]
        precision_rev = precision_lst[::-1]
        max_index = clip_when_decreasing(recall_rev)
        recall_rev, precision_rev = recall_rev[:max_index], precision_rev[:max_index]
        f = interp1d(recall_rev, precision_rev, assume_sorted=True, bounds_error=False, fill_value=(precision_rev[0], precision_rev[-1]))  
        y_rev = f(OMNIRECALL)
        precision = y_rev[::-1]
        color = 'tab:blue' if i == best_seeds[pu_config] else "gray"
        ax.plot(x[27:-27], precision[27:-27], color=color, label=seed_id)
    ax.grid()
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    fig.savefig(f"./figures/supp_figures/SDPR_per_seed_{pu_config}.png", bbox_inches='tight')
    plt.close(fig)

def main():
    for key in UNet_025.keys():
        aucs = get_AUPR(key, UNet_025[key])
        max_auc = aucs.index(max(aucs))
        print("\n***************************")
        print(f"Max index for {key} = {max_auc}")
        print("***************************\n")
        # plot_seeds(key, UNet_025[key])


if __name__=="__main__":
    main()
