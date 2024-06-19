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

def clip_when_decreasing(recall):
    max_index = recall.index(max(recall))
    return max_index

def compute_f1(recall, precision):
    epsilon = 1e-5 if recall+precision == 0 else 0
    return (2*recall*precision)/(recall+precision+epsilon)

def aggregate_seeds(models):
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
        precision[i] = y_rev[::-1]
        temp_p = y_rev[::-1]
        temp_r = OMNIRECALL[::-1]
        f1 = [compute_f1(r, p) for r, p in zip(temp_r, temp_p)]
        max_f1 = max(f1)
        max_idx = f1.index(max(f1))
        if max_f1 > best_f1:
            best_f1 = max_f1
            max_r = temp_r[max_idx]
            max_p = temp_p[max_idx]
    return precision, (max_r, max_p)

def aggregate_stardist_seeds(models, pu_config):
    recall = np.zeros((len(models), 100))
    precision = np.zeros((len(models), 100))
    best_f1 = 0
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
        precision[i] = y_rev[::-1]
    return precision

def main_unet():
    # colors = ["black", "tab:green", "tab:red", "tab:blue", "tab:purple", "tab:orange", "tab:pink", "tab:olive", "tab:brown"]
    # colors = ["tab:blue"] + ["tab:orange"] * 8
    # alphas = [0.3, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    labels = ["4-0", "1-0", "1-1", "1-2", "1-4", "1-8", "1-16", "1-32", "1-64", "1-128", "1-256"]
    # colors = ["black", "tab:green", "tab:red", "tab:blue", "tab:orange"]
    # labels = ['4-0', '4-1', '4-2', '4-4', '4-8']

    cm = plt.get_cmap('nice-prism')
    NUM_COLORS = len(labels) - 1
    # colors = [(0, 0, 0, 1 * alpha) for alpha in np.linspace(0.1, 1.0, len(labels))]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_prop_cycle(color=[cm(i/NUM_COLORS) for i in range(NUM_COLORS)])
    # colors = ['tab:blue', "tab:green", "tab:red", "tab:blue", "tab:orange", "tab:olive", "tab:purple", "black", "gold", "magenta", "cyan"] # just to visualize a bit better exactly which ones are best
    x = OMNIRECALL[::-1]
    for i, model_config in enumerate(tqdm(UNet_025.keys(), desc="UNet models...")):
        # recall, precision = aggregate_seeds(UNet_025[model_config])
        precision, _ = aggregate_seeds(UNet_025[model_config])
        precision_mean = np.mean(precision, axis=0)
        precision_std = np.std(precision, axis=0)
        # recall_mean, precision_mean = np.mean(recall, axis=0), np.mean(precision, axis=0)
        # recall_std, precision_std = np.std(recall, axis=0), np.mean(precision, axis=0)
        if i == 0:
            ax.plot(x[27:-10], precision_mean[27:-10], color='black', label=labels[i], ls='--')
        else:
            ax.plot(x[27:-10], precision_mean[27:-10], label=labels[i])
    ax.legend()
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim([0.0, 0.8])
    ax.set_ylim([0.0, 0.85])
    plt.grid(True)
    fig.savefig("./figures/PR_curves/UNet025_PR.png", bbox_inches="tight")
    fig.savefig("./figures/PR_curves/UNet025_PR.pdf", bbox_inches='tight', transparent=True)
    plt.close(fig)
    print("******** DONE *********")

def main_stardist():
    labels = ["4-0", "1-0", "1-1", "1-2", "1-4", "1-8", "1-16", "1-32", "1-64", "1-128", "1-256"]
    cm = plt.get_cmap('nice-prism')
    NUM_COLORS = len(labels) - 1
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_prop_cycle(color=[cm(i/NUM_COLORS) for i in range(NUM_COLORS)])
    x = OMNIRECALL[::-1]
    for i, model_config in enumerate(tqdm(STARDIST_025, desc="StarDist models...")):
            # recall, precision = aggregate_seeds(UNet_025[model_config])
            precision = aggregate_stardist_seeds(STARDIST_025[model_config], pu_config=model_config)
            precision_mean = np.mean(precision, axis=0)
            precision_std = np.std(precision, axis=0)
            # ax.plot(recall_mean[6:], precision_mean[6:], label=labels[i], color=colors[i])
            # ax.plot(x[27:-27], precision_mean[27:-27], label=labels[i], color=colors[i])
            # ax.fill_between(x[27:-27], precision_mean[27:-27]-precision_std[27:-27], precision_mean[27:-27]+precision_std[27:-27], color=colors[i], alpha=0.3)
            if i ==0:
                ax.plot(x[22:-5], precision_mean[22:-5], color='black', label=labels[i], ls='--')
            else:
                ax.plot(x[22:-5], precision_mean[22:-5], label=labels[i])
            # ax.scatter(max_coords[0], max_coords[1], color=colors[i], edgecolors='black', marker='*', s=200)
    ax.legend()
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim([0.0, 0.8])
    ax.set_ylim([0.0, 0.85])
    plt.grid(True)
    fig.savefig("./figures/PR_curves/StarDist25_PR.png", bbox_inches="tight")
    fig.savefig("./figures/PR_curves/StarDist25_PR.pdf", bbox_inches='tight', transparent=True)
    plt.close(fig)
    print("Permission was denied for the following files:")
    for p in permission_denied:
        print(p)
    print("The following files were not found:")
    for f in file_not_found:
        print(f)
    print("******** DONE *********")


def main():
    if args.backbone == "UNet":
        main_unet()
    elif args.backbone == "StarDist":
        main_stardist()
    else:
        exit("The requested backbone model does not exist. Use UNet or StarDist.")
   

if __name__=="__main__":
    main()
