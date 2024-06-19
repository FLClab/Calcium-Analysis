import argparse
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import tifffile
import pandas
from typing import Tuple
from matplotlib import patches

parser = argparse.ArgumentParser()
parser.add_argument("--file", type=str, default="34-1")
parser.add_argument("--num_pos", type=int, default=8)
args = parser.parse_args()
GROUND_TRUTH_PATH = "/home/frbea320/projects/def-flavielc/frbea320/mscts-analysis/data/testset"


expert_file = f"{GROUND_TRUTH_PATH}/manual-expert/manual_{args.file}.csv"

def get_movie(movie: str) -> np.ndarray:
    movie_path = f"{GROUND_TRUTH_PATH}/raw-input/{movie}.tif"
    return tifffile.imread(movie_path)

def get_expert_centroids() -> Tuple[np.ndarray, np.ndarray]:
    fname = expert_file.split("/")[-1].split(".")[0].split("_")[-1]
    movie = get_movie(fname)
    data = pandas.read_csv(expert_file)
    truth_centroids = np.stack((data["Slice"], data["Y"], data["X"])).T
    return truth_centroids, movie

def generate_pu_example(centroids: np.ndarray, movie: np.ndarray) -> None:
    indices = np.arange(centroids.shape[0])
    indices = np.random.choice(indices, size=args.num_pos)
    centroids = centroids[indices]
    movie_maxproj = np.max(movie, axis=0)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(movie_maxproj, cmap='Greys', vmax=0.35*np.max(movie_maxproj)) # 0.35 seems to be the right multiplier for the default file (34-1)
    for c in centroids:
        print(c)
        y, x = int(c[1]), int(c[2])
        ymax, xmax, ymin, xmin = y+16, x+16, y-16, x-16
        rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax-ymin, lw=3, edgecolor='tab:green', facecolor='none')
        ax.add_patch(rect)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.savefig(f"./figures/pu_examples/{args.file}.png", bbox_inches='tight')
    fig.savefig(f"./figures/pu_examples/{args.file}.pdf", bbox_inches='tight', transparent=True)


def main():
    centroids, movie = get_expert_centroids()
    print(centroids.shape)
    generate_pu_example(centroids=centroids, movie=movie)

if __name__=="__main__":
    main()