import numpy as np
import matplotlib.pyplot as plt
from model_paths import STARDIST_025
import pandas
import pickle
import seaborn
import matplotlib
from scipy.stats.kde import gaussian_kde
from scipy.stats import norm
from typing import List
import argparse
import random

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="UNet")
parser.add_argument("--aggregate", action="store_true")
parser.add_argument("--no-aggregate", dest="aggregate", action="store_false")
args = parser.parse_args()

cmap_og = matplotlib.colors.LinearSegmentedColormap.from_list(
    name="nice-prism",
    colors=["#5F4690","#1D6996","#38A6A5","#0F8554","#73AF48","#EDAD08","#E17C05","#CC503E","#94346E"]
)
matplotlib.colormaps.register(cmap=cmap_og, force=True)
matplotlib.colormaps.register(cmap=cmap_og.reversed(), force=True)

model_keys = ["4-0", "1-0", "1-1", "1-2", "1-4", "1-8", "1-16", "1-32", "1-64", "1-128", "1-256"]
event_keys = ["onSynapse", "onDendrite", "smallArea", "bigArea", "outOfFocus", "longArea", "highIntensity", "lowIntensity"]

PEAK_DICE = 0.7354709418837675


SD_4_0 = 4
SD_1_0 = 9
SD_1_1 = 2
SD_1_2 = 23
SD_1_4 = 16
SD_1_8 = 23
SD_1_16 = 8
SD_1_32 = 9
SD_1_64 = 6
SD_1_128 = 19
SD_1_256 = 5

model_dict = {1: '4-0',
              2: '1-0',
              3: '1-1',
              4: '1-2',
              5: '1-4',
              6: '1-8',
              7: '1-16',
              8: '1-32',
              9: '1-64',
              10: '1-128',
              11: '1-256',
              }

STARDIST_PATHS = [
    f"/home/frbea320/scratch/StarDist3D/{STARDIST_025['4-0'][SD_4_0]}",
    f"/home/frbea320/projects/def-flavielc/frbea320/mscts-analysis/data/testset/StarDist3D/{STARDIST_025['1-0'][SD_1_0]}",
    f"/home/frbea320/projects/def-flavielc/frbea320/mscts-analysis/data/testset/StarDist3D/{STARDIST_025['1-1'][SD_1_1]}",
    f"/home/frbea320/projects/def-flavielc/frbea320/mscts-analysis/data/testset/StarDist3D/{STARDIST_025['1-2'][SD_1_2]}",
    f"/home/frbea320/projects/def-flavielc/frbea320/mscts-analysis/data/testset/StarDist3D/{STARDIST_025['1-4'][SD_1_4]}",
    f"/home/frbea320/projects/def-flavielc/frbea320/mscts-analysis/data/testset/StarDist3D/{STARDIST_025['1-8'][SD_1_8]}",
    f"/home/frbea320/projects/def-flavielc/frbea320/mscts-analysis/data/testset/StarDist3D/{STARDIST_025['1-16'][SD_1_16]}",
    f"/home/frbea320/projects/def-flavielc/frbea320/mscts-analysis/data/testset/StarDist3D/{STARDIST_025['1-32'][SD_1_32]}",
    f"/home/frbea320/projects/def-flavielc/frbea320/mscts-analysis/data/testset/StarDist3D/{STARDIST_025['1-64'][SD_1_64]}",
    f"/home/frbea320/projects/def-flavielc/frbea320/mscts-analysis/data/testset/StarDist3D/{STARDIST_025['1-128'][SD_1_128]}",
    f"/home/frbea320/projects/def-flavielc/frbea320/mscts-analysis/data/testset/StarDist3D/{STARDIST_025['1-256'][SD_1_256]}"
]

cmap = plt.get_cmap('nice-prism', len(STARDIST_PATHS))

model_keys = ["4-0", "1-0", "1-1", "1-2", "1-4", "1-8", "1-16", "1-32", "1-64", "1-128", "1-256"]

def load_data(model: str) -> dict:
    path = "./paper_figures/UNet_eventtype_theresa_segmentation.pkl" if model == "UNet" else "./paper_figures/StarDist_eventtype_theresa2_segmentation.pkl"
    data = pickle.load(open(path, "rb"))
    return data

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
        max_density.append(xx[idx])
        curves.append(y + curve[idx])
        print(y)
        print(xx[idx])
        if fill:
            c = "black" if i == 0 else cmap(i-1)
            plt.fill_between(xx, np.ones(n_points)*y, 
                             curve+y, zorder=len(data)-i+1, color=c)
        plt.plot(xx, curve+y, c='white', zorder=len(data)-i+1)
        plt.axvline(x=xx[idx], ymin=y, ymax=y+curve[idx], color='white', ls='--', zorder=len(data)-i+1)
    # plt.axhline(y=0, xmin=xx[0], xmax=xx[-1], ls='dotted', lw=10, color='gray')
    if labels:
        plt.yticks(ys, labels)
    return max_density

    
def plot_ridgelines(data: List[list], labels: list, key: str, num: float) -> None:
    fig = plt.figure(figsize=(8, 10))

    max_density = ridgeline(data, labels=labels, overlap=.15, fill='tomato')
    plt.title('Dice scores per model', loc='left', fontsize=18, color='gray')
    plt.gca().spines['left'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.xlabel('Dice')
    plt.xlim([0.0, 1.0])
    plt.grid(zorder=0)
    # fig.savefig(f"./ridgeline_plots/theresa_1/{args.model}_{key}_ridgeline_{num}events.png")
    fig.savefig(f"./ridgeline_plots/theresa_1/{args.model}_{key}_ridgeline_{num}events_median.pdf", bbox_inches='tight', transparent=True)
    return max_density

def aggregate_results(data) -> List[list]:
    results = {
        key: [] for key in model_keys
    }
    for mkey in model_keys:
        for ekey in event_keys:
            new_data = data[ekey][mkey]
            results[mkey] += new_data
    return results

def main():
    data = load_data(model=args.model)
    if args.aggregate:
        for event_key in data.keys():
            d = data[event_key]
            num_events = [len(d[mkey]) for mkey in model_keys]
            num_mean = round(np.mean(np.array(num_events)), 3)
            data_list = [d[mkey] for mkey in model_keys]
            plot_ridgelines(data=data_list, labels=model_keys, key=event_key, num=num_mean)
    else:
        data = aggregate_results(data)
        num_events = [len(data[mkey]) for mkey in model_keys]
        num_mean = round(np.mean(np.array(num_events)), 3)
        data_list = [data[mkey] for mkey in model_keys]
        mean_values = [np.mean(data[mkey]) for mkey in model_keys]
        min_values = [np.min(data[mkey]) for mkey in model_keys]
        max_values = [np.max(data[mkey]) for mkey in model_keys]
        median_values = [np.median(data[mkey]) for mkey in model_keys]
        perc10 = [np.percentile(data[mkey], 10) for mkey in model_keys]
        perc90 = [np.percentile(data[mkey], 90) for mkey in model_keys]
        max_density = plot_ridgelines(data=data_list, labels=model_keys, key="all", num=num_mean)
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
        df.to_csv(f'{args.model}_dice_distribution_stats.csv', index=False)
        


if __name__=="__main__":
    main()