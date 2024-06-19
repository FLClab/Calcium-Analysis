import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import skimage.measure
from tqdm import tqdm
import tifffile
import argparse
import glob
import seaborn
from itertools import combinations

GROUND_TRUTH_PATH = "/home/frbea320/projects/def-flavielc/frbea320/mscts-analysis/data/testset"
PREDICTION_PATH = "/home/frbea320/projects/def-flavielc/frbea320/mscts-analysis/data/testset/StarDist3D/detection"

# TODO: aspect ratio as color code
FEATURES = ["Area", "Mean intensity", "Duration", "Aspect ratio", "Solidity"]

feature_pairs = list(combinations(FEATURES, 2))
model_name = "StarDist3D_complete_1-4_46"

def load_data(csv_file="./features_dataframe.csv"):
    df = pd.read_csv(csv_file)
    return df

def feature_graphs(df):
    custom_palette = ["magenta", "r", "gold", "blue"]
    for pair in feature_pairs:
        feature1, feature2 = pair
        use_log = False
        if feature1 == "Area":
            use_log = True
        print(feature1, feature2)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        im = seaborn.scatterplot(ax=ax, data=df, x=feature1, y=feature2, hue="Cluster", alpha=0.7, palette=seaborn.color_palette(custom_palette, 4))
        if use_log:
            ax.set_xscale('log')
        fig.savefig(f"./feature_graphs_2d/{feature1}-vs-{feature2}.png", bbox_inches="tight")
        fig.savefig(f"./feature_graphs_2d/{feature1}-vs-{feature2}.pdf", bbox_inches='tight', transparent=True)
        plt.close(fig)


def main():
    df = load_data()
    feature_graphs(df)
    
if __name__=="__main__":
    main()
