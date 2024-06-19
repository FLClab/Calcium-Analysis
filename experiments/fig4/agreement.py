import numpy as np
import matplotlib.pyplot as plt
import pickle
import tifffile
from model_paths import UNet_025, STARDIST_025
from tqdm import tqdm  
import glob
# from metrics.segmentation import commons
#from dice_vs_centroid import load_data
import argparse
from scipy.stats.kde import gaussian_kde
from typing import List
import pandas


parser = argparse.ArgumentParser()
parser.add_argument("--expert", type=str, default="theresa_1")
parser.add_argument("--compute", action="store_true")
parser.add_argument("--no-compute", dest="compute", action="store_false")
args = parser.parse_args()

UNET_MODEL = f"/home/frbea320/projects/def-flavielc/frbea320/mscts-analysis/data/baselines/ZeroCostDL4Mic/{UNet_025['UNet3D_complete_1-0'][2]}"
STARDIST_MODEL = f"/home/frbea320/projects/def-flavielc/frbea320/mscts-analysis/data/testset/StarDist3D/{STARDIST_025['1-0'][9]}"

MINIFINDER_PATH = "/home/frbea320/projects/def-flavielc/frbea320/mscts-analysis/data/testset/minifinder"
FLAVIE1_PATH = "/home/frbea320/projects/def-flavielc/frbea320/mscts-analysis/data/testset/segmentation-expert/mask_flavie_1"
FLAVIE2_PATH = "/home/frbea320/projects/def-flavielc/frbea320/mscts-analysis/data/testset/segmentation-expert/mask_flavie_2"
THERESA1_PATH = "/home/frbea320/projects/def-flavielc/frbea320/mscts-analysis/data/testset/segmentation-expert/mask_theresa_1"
THERESA2_PATH = "/home/frbea320/projects/def-flavielc/frbea320/mscts-analysis/data/testset/segmentation-expert/mask_theresa_2"
RAW_PATH = "/home/frbea320/projects/def-flavielc/frbea320/mscts-analysis/data/testset/raw-input"

model_keys = ["ITD", "StarDist", "UNet", "Inter", "Intra"]

def get_expert_path() -> str:
    if args.expert == "flavie_1":
        return FLAVIE1_PATH
    elif args.expert == "flavie_2":
        return FLAVIE2_PATH
    elif args.expert == "theresa_1":
        return THERESA1_PATH
    else:
        return THERESA1_PATH
    
def get_expert_agreement() -> str:
    if args.expert == "flavie_1":
        data = pickle.load(open("./paper_figures/agreement_flavie_1.pkl", "rb"))
    elif args.expert == "flavie_2":
        data = pickle.load(open("./paper_figures/agreement_flavie_2.pkl", "rb"))
    elif args.expert == "theresa_1":
        data = pickle.load(open("./paper_figures/agreement_theresa_1.pkl", "rb"))
    else:
        data = pickle.load(open("./paper_figures/agreement_theresa_2.pkl", "rb"))
    return data


def compare_models(data: dict) -> dict:
    results = {
        "ITD": [],
        "StarDist": [],
        "UNet": [],
        "Inter": [],
        "Intra": []
    }
    for _, values in tqdm(data.items()):
        for movie_id, coords, time_slice, index in tqdm(values, desc="Events..."):
            intra_expert = FLAVIE2_PATH if args.expert == "flavie_1" else THERESA2_PATH
            inter_expert = THERESA1_PATH if args.expert == "flavie_1" else FLAVIE1_PATH
            expert_path = get_expert_path()
            truth_name = glob.glob(f"{expert_path}/*-{index}.tif")[0]
            intra_name = glob.glob(f"{intra_expert}/*-{index}.tif")[0]
            inter_name = glob.glob(f"{inter_expert}/*-{index}.tif")[0]
            truth = tifffile.imread(truth_name)
            intra = tifffile.imread(intra_name)
            inter = tifffile.imread(inter_name)
            x, y, width, height = coords
            stardist_path = f"{STARDIST_MODEL}/{movie_id}".replace(".tif", ".npz")
            unet_path = f"{UNET_MODEL}/Quality Control Segmentation/Prediction/{movie_id}".replace(".tif", "_prediction.tif")
            stardist_pred = np.load(stardist_path)['label']
            unet_pred = tifffile.imread(unet_path)
            stardist_pred = stardist_pred[time_slice, y:y+height, x:x+width]
            unet_pred = unet_pred[time_slice, y:y+height, x:x+width]
            minifinder = np.load(f"{MINIFINDER_PATH}/{movie_id}".replace(".tif", ".npy"))
            minifinder_pred = minifinder[time_slice, y:y+height, x:x+width]
            minifinder_dice = commons.dice(truth, minifinder_pred)
            stardist_dice = commons.dice(truth, stardist_pred)
            unet_dice = commons.dice(truth, unet_pred)
            inter_dice = commons.dice(truth, inter)
            intra_dice = commons.dice(truth, intra)
            results["ITD"].append(minifinder_dice)
            results["StarDist"].append(stardist_dice)
            results["UNet"].append(unet_dice)
            results["Inter"].append(inter_dice)
            results["Intra"].append(intra_dice)
    return results

def ridgeline(data, overlap=0, fill=True, labels=None, n_points=500):
    if overlap > 1 or overlap < 0:
        raise ValueError('overlap must be in [0 1]')
    xx = np.linspace(0, 1, n_points)
    curves = []
    ys = []
    max_density = []
    for i, d in enumerate(data):
        pdf = gaussian_kde(d)
        y = i*(1.5-overlap)
        ys.append(y)
        curve = pdf(xx)
        idx = np.argmax(curve)
        max_x = xx[idx]
        max_density.append(max_x)
        if fill:
            plt.fill_between(xx, np.ones(n_points)*y, 
                             curve+y, zorder=len(data)-i+1, color=fill)
        plt.plot(xx, curve+y, c='white', zorder=len(data)-i+1)
        if labels[i] == "Inter":
            plt.axvline(x=max_x, ymin=np.min(curve), ymax=np.max(curve), color='black', ls='--')
    if labels:
        plt.yticks(ys, labels)
    return max_density

    
def plot_ridgelines(data: List[list], labels: list) -> None:
    fig = plt.figure()
    max_density = ridgeline(data, labels=labels, overlap=.15, fill='gray')
    plt.title('Dice scores per model', loc='left', fontsize=18, color='gray')
    plt.gca().spines['left'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.xlabel('ppm')
    plt.xlim([0.0, 1.0])
    plt.grid(zorder=0)
    # fig.savefig(f"./ridgeline_plots/theresa_1/{args.model}_{key}_ridgeline_{num}events.png")
    fig.savefig(f"./ridgeline_plots/{args.expert}/models_agreement_ridgeline.pdf", bbox_inches='tight', transparent=True)
    return max_density


def main():
    if args.compute:
        # data = load_data()
        results = compare_models(data=data)
        with open(f"./paper_figures/agreement_{args.expert}.pkl", "wb") as f:
            pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        data = get_expert_agreement()
        keys = list(data.keys())
        data_list = [data[key] for key in data.keys()]
        mean_values = [np.mean(data[mkey]) for mkey in model_keys]
        min_values = [np.min(data[mkey]) for mkey in model_keys]
        max_values = [np.max(data[mkey]) for mkey in model_keys]
        median_values = [np.median(data[mkey]) for mkey in model_keys]
        perc10 = [np.percentile(data[mkey], 10) for mkey in model_keys]
        perc90 = [np.percentile(data[mkey], 90) for mkey in model_keys]
        max_density = plot_ridgelines(data=data_list, labels=keys)
        df = pandas.DataFrame(columns=["Model", "Min", "Max", "Median", "Mean", "10th percentile", "90th percentile", "Max density"])
        for i in range(len(max_density)):
            data = []
            data.append(model_keys[i])
            data.append(min_values[i])
            data.append(max_values[i])
            data.append(median_values[i])
            data.append(mean_values[i])
            data.append(perc10[i])
            data.append(perc90[i])
            data.append(max_density[i])
            df.loc[i] = data
        df.to_csv(f'{args.expert}_dice_distribution_stats.csv', index=False)

if __name__=="__main__":
    main()