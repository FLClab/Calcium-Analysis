import numpy as np
import matplotlib.pyplot as plt
import pandas
from model_paths import UNet_025
import pickle
import seaborn

BASE_PATH = f"/home/frbea320/projects/def-flavielc/frbea320/mscts-analysis/data/baselines/ZeroCostDL4Mic"

FEATURES = [
    "area",
    "axis_major_length",
    "axis_minor_length",
    "intensity_max",
    "intensity_min",
    "solidity"
    ]

FEATURE_DICT = { 
    task: {
        key: [] for key in FEATURES
    } for task in ["Detection", "Segmentation"]
}
def populate_feature_dict(model, task="Detection"):
    path = f"{BASE_PATH}/{model}/Quality Control {task}/QC_regionprops_{model}.pkl"
    data = pickle.load(open(path, "rb"))
    for video_id in data.keys():
        video = data[video_id]
        video_keys = video.keys()
        if any([item not in video_keys for item in FEATURES]):
            continue
        for feature in FEATURES:
            feature_values = video[feature].tolist()
            FEATURE_DICT[task][feature] += feature_values

def aggregate_seeds(models):
    for seed in models:
        try:
            populate_feature_dict(seed, "Detection")
        except:
            print("Skipping detection for {}".format(seed))
        try:
            populate_feature_dict(seed, "Segmentation")
        except:
            print("Skipping segmentation for {}".format(seed))

def plot_feature_distributions(feature_dict=FEATURE_DICT):
    detection_df = pandas.DataFrame.from_dict(feature_dict["Detection"])
    segmentation_df = pandas.DataFrame.from_dict(feature_dict["Segmentation"])
    detection_labels = ["Detection"] * detection_df.shape[0]
    detection_df["Task"] = detection_labels
    segmentation_labels = ["Segmentation"] * segmentation_df.shape[0]
    segmentation_df["Task"] = segmentation_labels
    df = pandas.concat([detection_df, segmentation_df])
    for feature in FEATURES:
        upper = np.percentile(df[feature], 98)
        lower = np.percentile(df[feature], 2)
        df[feature] = df[feature].clip(lower=lower, upper=upper)
        fig = plt.figure()
        ax = seaborn.violinplot(data=df, x="Task", y=feature, hue="Task")
        plt.savefig("./paper_figures/features/{}.png".format(feature), bbox_inches="tight")
        plt.close(fig)

def main():
    for pu_config in UNet_025.keys():
        aggregate_seeds(UNet_025[pu_config])

    df = plot_feature_distributions()

    

if __name__=="__main__":
    main()