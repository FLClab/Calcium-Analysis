import numpy as np
import pandas 
import seaborn
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

FULL_DIST_WASSERSTEIN = {
    "UNet_4-0": 75.25274595751513,
    "UNet_1-0": 104.3824987513226,
    "UNet_1-64": 86.1322712732886,
    "StarDist_4-0": 37.681733810678566,
    "StarDist_1-0": 40.736504507663085,   
}

def load_features():
    df = pandas.read_csv("./model_features.csv")
    return df

def feature_violin(df: pandas.DataFrame):
    for i in range(df.shape[1]):
        feature = df.columns[i]
        print(feature)
        if feature in ["model", "Unnamed: 0"]:
            continue
        fig = plt.figure()
        seaborn.violinplot(data=df, x='model', y=feature)
        fig.savefig(f"./feature_distributions/{feature}_violin.png")
        plt.close(fig)

def feature_bar(df:pandas.DataFrame):
    gt = df[df["model"] == "Ground truth"]
    u4 = df[df["model"] == "UNet_4-0"]
    u1 = df[df["model"] == "UNet_1-0"]
    u64 = df[df["model"] == "UNet_1-64"]
    s4 = df[df["model"] == "StarDist_4-0"]
    s1 = df[df["model"] == "StarDist_1-0"]
    for i in range(u4.shape[1]):
        feature = u4.columns[i]
        if feature in ["model", "Unnamed: 0"]:
            continue
        gt_vals = gt.iloc[:, i].to_numpy()
        u4_vals = u4.iloc[:, i].to_numpy()
        u1_vals = u1.iloc[:, i].to_numpy()
        u64_vals = u64.iloc[:, i].to_numpy()
        s4_vals = s4.iloc[:, i].to_numpy()
        s1_vals = s1.iloc[:, i].to_numpy()
        u4_diff = np.mean(gt_vals) - np.mean(u4_vals)
        u1_diff = np.mean(gt_vals) - np.mean(u1_vals)
        u64_diff = np.mean(gt_vals) - np.mean(u64_vals)
        s4_diff = np.mean(gt_vals) - np.mean(s4_vals)
        s1_diff = np.mean(gt_vals) - np.mean(s1_vals)
        fig = plt.figure()
        x = np.arange(0, 5, 1)
        data = [u4_diff, u1_diff, u64_diff, s4_diff, s1_diff]
        plt.bar(x, data, width=0.5)
        plt.axhline(0, x[0], x[-1] + 0.5, color='black', ls='--')
        plt.xticks(x, ["U_4-0", "U_1-0", "U_1-64", "SD_4-0", "SD_1-0"])
        plt.ylabel(f"{feature} difference w/\nground truth")
        plt.tight_layout()
        fig.savefig(f"./feature_distributions/{feature}_bar.png")
        plt.close(fig)

def main():
    features = load_features()
    feature_violin(features)
    feature_bar(features)

if __name__=="__main__":
    main()