import numpy as np
import matplotlib.pyplot as plt
import pandas
from model_paths import UNet_025
import pickle
import seaborn

THRESHOLD_PATH = "/home/frbea320/projects/def-flavielc/frbea320/mscts-analysis/data/baselines/ZeroCostDL4Mic"
PICKLE_PATH = "/home/frbea320/projects/def-flavielc/frbea320/mscts-analysis/data/testset/UNet3D"

def get_thresholds(model):
    det_path = f"{THRESHOLD_PATH}/{model}/optimized-threshold-detection"
    seg_path = f"{THRESHOLD_PATH}/{model}/optimized-threshold-segmentation"
    with open(det_path, "r") as f:
        det_threshold = float(f.read())
    with open(seg_path, "r") as f:
        seg_threshold = float(f.read())
    return det_threshold, seg_threshold

def get_closest_label(threshold, label_keys):
    closest = min(label_keys, key=lambda x:abs(x-threshold))
    label_keys = label_keys.tolist()
    return label_keys.index(closest)

def get_pickle_results(model):
    d_threshold, s_threshold = get_thresholds(model)
    pkl_data = pickle.load(open(f"{PICKLE_PATH}/{model}/segmentation_results.pkl", "rb"))
    thresholds = pkl_data["thresholds"]
    d_threshold_idx = get_closest_label(d_threshold, thresholds)
    s_threshold_idx = get_closest_label(s_threshold, thresholds)
    dice_data = pkl_data["dice"]
    d_dice = dice_data[d_threshold_idx]
    s_dice = dice_data[s_threshold_idx]
    return d_dice, s_dice

def aggregate_seeds(models):
    det_data, seg_data = [], []
    for seed in models:
        d_dice, s_dice = get_pickle_results(seed)
        det_data.append(d_dice)
        seg_data.append(s_dice)
    return det_data, seg_data

def detection_vs_segmentation_dataframe(*args):
    df = pandas.DataFrame(columns=["Dice", "Optimized for"])
    for i, data in enumerate(args):
        dice_detection, dice_segmentation = data[0], data[1]
        detection_labels = np.array(["Detection"] * len(dice_detection))
        segmentation_labels = np.array(["Segmentation"] * len(dice_segmentation))
        dice_detection = np.array(dice_detection)
        detection_data = np.c_[dice_detection, detection_labels]
        df_detection = pandas.DataFrame(detection_data, columns=["Dice", "Optimized for"])
        df = pandas.concat([df, df_detection], ignore_index=True)
        dice_segmentation = np.array(dice_segmentation)
        segmentation_data = np.c_[dice_segmentation, segmentation_labels]
        df_segmentation = pandas.DataFrame(segmentation_data, columns=["Dice", "Optimized for"])
        df = pandas.concat([df, df_segmentation], ignore_index=True)
    return df
        
def seaborn_bar(df):
    print(df["Dice"].max(), df["Dice"].min())
    yticks = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    ylabels = [str(item) for item in yticks]
    ax = seaborn.boxplot(data=df, x="Optimized for", y="Dice", hue="Optimized for")
    #ax = seaborn.swarmplot(data=df, x="Optimized for", y="Dice", color="grey", alpha=0.6)
    plt.yticks(ticks=yticks, labels=ylabels)
    plt.ylim([0.0, 1.0])
    plt.savefig("./seaborn-segmentation-barplot.png", bbox_inches='tight')

def main():
    data_1_0 = aggregate_seeds(UNet_025["UNet3D_complete_1-0"])
    data_25_0 = aggregate_seeds(UNet_025["25-0"])
    data_25_1 = aggregate_seeds(UNet_025["25-1"])
    data_25_2 = aggregate_seeds(UNet_025["25-2"])
    data_25_4 = aggregate_seeds(UNet_025["25-4"])
    data_25_8 = aggregate_seeds(UNet_025["25-8"])
    df = detection_vs_segmentation_dataframe(
        data_1_0,
        data_25_0,
        data_25_1,
        data_25_2,
        data_25_4,
        data_25_8,
    )
    df["Dice"] = pandas.to_numeric(df["Dice"])
    seaborn_bar(df)

if __name__=="__main__":
    main()