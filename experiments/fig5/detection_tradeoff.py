import numpy as np
import matplotlib.pyplot as plt
import pandas 
from model_paths import UNet_025
import seaborn

BASE_PATH = f"/home/frbea320/projects/def-flavielc/frbea320/mscts-analysis/data/baselines/ZeroCostDL4Mic"

def get_results(model, task="Detection"):
    path = f"{BASE_PATH}/{model}/Quality Control {task}/QC_metrics_{model}.csv"
    df = pandas.read_csv(path)
    df = df[df["det-precision"] != -1.0]
    precision = df["det-precision"].mean(axis=0)
    recall = df["det-recall"].mean(axis=0)
    f1 = df["det-f1-score"].mean(axis=0)
    data = [recall, precision]
    return data

def aggregate_seeds(models):
    det_data = np.zeros((len(models), 2))
    seg_data = np.zeros((len(models), 2))
    for i, seed in enumerate(models):
        detection_data = get_results(seed, "Detection")
        segmentation_data = get_results(seed, "Segmentation")
        det_data[i] = detection_data
        seg_data[i] = segmentation_data
    return det_data, seg_data

def scatter_all_data(*args):
    fig = plt.figure()
    colors = ["tab:blue", "tab:orange", "tab:purple", "tab:green", "tab:cyan", "tab:red"]
    labels = ["4-0", "1-0", "1-1", "1-2", "1-4", "1-8"]
    assert len(colors) == len(args)
    for i, data in enumerate(args):
        print(data[0].shape, data[1].shape)
        exit()
        detection, segmentation = data[0], data[1]
        print(detection.shape, segmentation.shape)
        recall_detection, precision_detection = detection[:, 0], detection[:, 1]
        recall_segmentation, precision_segmentation = segmentation[:, 0], segmentation[:, 1]
        plt.scatter(recall_detection, precision_detection, color=colors[i], label=labels[i], marker='x')
        plt.scatter(recall_segmentation, precision_segmentation, color=colors[i], marker='o')
    plt.xlim([0.0, 1.0])
    plt.yticks([0.0, 1.0])
    plt.xlabel('Recall')
    plt.ylabel("Precision")
    plt.legend()
    fig.savefig("./scatterplot_all.png", bbox_inches='tight')

def seaborn_scatter(df):
    ax = seaborn.jointplot(data=df, x="Recall", y="Precision", hue="Optimized for")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.savefig("./seaborn-jointplot.png", bbox_inches='tight')

def detection_vs_segmentation_dataframe(*args):
    df = pandas.DataFrame(columns=["Recall", "Precision", "Optimized for"])
    for i, data in enumerate(args):
        detection_array = data[0]
        detection_labels = np.array(["Detection"] * detection_array.shape[0])
        detection_data = np.c_[detection_array, detection_labels]
        detection_df = pandas.DataFrame(detection_data, columns=["Recall", "Precision", "Optimized for"])
        df = pandas.concat([df, detection_df], ignore_index=True)  
        segmentation_array = data[1]
        segmentation_labels = np.array(["Segmentation"] * segmentation_array.shape[0])
        segmentation_data = np.c_[segmentation_array, segmentation_labels]
        segmentation_df = pandas.DataFrame(segmentation_data, columns=["Recall", "Precision", "Optimized for"])
        df = pandas.concat([df, segmentation_df], ignore_index=True)
    return df


def main():
    data_1_0 = aggregate_seeds(UNet_025["UNet3D_complete_1-0"])
    data_25_0 = aggregate_seeds(UNet_025["25-0"])
    data_25_1 = aggregate_seeds(UNet_025["25-1"])
    data_25_2 = aggregate_seeds(UNet_025["25-2"])
    data_25_4 = aggregate_seeds(UNet_025["25-4"])
    data_25_8 = aggregate_seeds(UNet_025["25-8"])
    # TODO:
    # data_25_16 = aggregate_seeds(UNet_025["25-16"])
    # data_25_32 = aggregate_seeds(UNet_025["25-32"])
    scatter_all_data(
        data_1_0,
        data_25_0,
        data_25_1,
        data_25_2,
        data_25_4,
        data_25_8
    )
    df = detection_vs_segmentation_dataframe(
        data_1_0,
        data_25_0,
        data_25_1,
        data_25_2,
        data_25_4,
        data_25_8
    )
    df["Recall"] = pandas.to_numeric(df["Recall"])
    df["Precision"] = pandas.to_numeric(df["Precision"])
    print(df.dtypes)
    seaborn_scatter(df)

if __name__=="__main__":
    main()
