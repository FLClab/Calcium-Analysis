import numpy as np
import matplotlib.pyplot as plt
import os
import pandas
import argparse
import tifffile
from sklearn.preprocessing import StandardScaler
import seaborn
from tqdm import tqdm
import skimage.measure
import sklearn.cluster
from sklearn.metrics import silhouette_score
import matplotlib
import glob

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True, type=str)
args = parser.parse_args()


GROUND_TRUTH_PATH = "/home/frbea320/projects/def-flavielc/frbea320/mscts-analysis/data/testset"
MODEL = f"/home/frbea320/projects/def-flavielc/frbea320/mscts-analysis/data/testset/StarDist3D/{args.model}"

FEATURES = [
    "Area",
    "Max intensity",
    "Duration",
    "Aspect ratio",
    "Solidity",
    "Frame",
    "X",
    "Y",
    "Video",
]

def collect_rprops():
    manual_expert_files = glob.glob(os.path.join(GROUND_TRUTH_PATH, "manual-expert", "*.csv"))
    vidnames = [fname.split("/")[-1].split(".")[0].split("_")[-1] for fname in manual_expert_files]
    all_rprops = {
        key: [] for key in vidnames
    }
    for vid, file in enumerate(tqdm(manual_expert_files, desc="Videos...")):
        data = pandas.read_csv(file)
        filename = os.path.basename(file)
        prediction_file = filename.split("_")[-1].split(".")[0]
        prediction_file = os.path.join(MODEL, prediction_file + ".npz")
        tail = prediction_file.split("/")[-1].replace(".npz", ".tif")
        input_file = f"{GROUND_TRUTH_PATH}/raw-input/{tail}"
        movie = tifffile.imread(input_file)
        pred_data = np.load(prediction_file)
        pred = pred_data["label"]
        pred_label = skimage.measure.label(pred)
        pred_rprops = skimage.measure.regionprops(pred_label, intensity_image=movie)
        all_rprops[vidnames[vid]] = pred_rprops
    return all_rprops

def compute_num_transients(rprops):
    num_transients = 0
    for item in rprops.items():
        num_transients += len(item[1])
    return num_transients

def create_dataframe(data, num_transients):
    num_features = len(FEATURES)
    big_ary = np.zeros((num_transients, num_features))
    counter = 0
    skipped = 0
    for item in data.items():
        fname = item[0]
        fname = float(fname.replace("-", "."))
        rprops = item[1]
        intensities = [r.mean_intensity for r in rprops]
        normalized_intensities = [r / max(intensities) for r in intensities]
        for i, r in enumerate(rprops):
            try:
                area = r.area
                t1, _, _,t2, _, _ = r.bbox
                t, y, x = r.weighted_centroid
                mean_i = normalized_intensities[i]
                major = r.major_axis_length
                minor = r.minor_axis_length
                aspect_ratio = major / minor if minor != 0 else 0
                solidity = r.solidity
                duration = t2 - t1
                big_ary[counter] = [
                    area,
                    mean_i,
                    duration,
                    aspect_ratio,
                    solidity,
                    t,
                    y,
                    x,
                    fname
                ]
                counter += 1
            except:
                skipped += 1
                print("Having to skip this region")
                continue
    print(big_ary.shape)
    big_ary = big_ary[:-skipped, :]
    df = pandas.DataFrame(big_ary, columns=FEATURES)
    return df

def kmeans_subtypes(df, num_clusters=list(range(2, 20))):
    feature_df = df.iloc[:, :-4]
    frames = df.iloc[:, -4]
    ys = df.iloc[:, -3]
    xs = df.iloc[:, -2]
    movie_id = df.iloc[:, -1]
    scaler = StandardScaler()
    feature_df = scaler.fit_transform(feature_df)
    kmeanses, clusterers, scores = [], [], []
    for nc in num_clusters:
        clusterer = sklearn.cluster.KMeans(n_clusters=nc, random_state=42)
        clusterers.append(clusterer)
        kmeans = clusterer.fit(feature_df)
        kmeanses.append(kmeans)
        silscore = silhouette_score(feature_df, kmeans.labels_)
        scores.append(silscore)
    fig = plt.figure()
    plt.plot(num_clusters, scores)
    fig.savefig("./silouhette_scores.png")
    return feature_df, scaler, clusterers[2], kmeanses[2], movie_id, frames, ys, xs

def reconstruct_points(points, scaler):
    return scaler.inverse_transform(points)

def compute_average_features_per_cluster(scaled_df, scaler, clusterer, kmeans, movie_id, frames, ys, xs):
    df = reconstruct_points(scaled_df, scaler)
    print('\nCOMPUTE_AVERAGE_FEATURES_PER_CLUSTER')
    print(df.shape)
    df = np.c_[df, frames, ys, xs, movie_id]
    df = np.c_[df, kmeans.labels_]
    print(df.shape)
    f_features = [
        "Area",
        "Mean intensity",
        "Duration",
        "Aspect ratio",
        "Solidity"
    ]
    df = pandas.DataFrame(df, columns=[
       "Area", 
        "Mean intensity", 
        "Duration", 
        "Aspect ratio",
        "Solidity",
        "Frame",
        "Y",
        "X",
        "Video",
        "Cluster", 
    ])
    df.to_csv("./features_dataframe.csv")
    for f in f_features:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        seaborn.kdeplot(data=df, x=f, hue="Cluster", ax=ax)
        fig.savefig(f"./kde_{f}.png")
        fig.savefig(f"./kde_{f}.pdf", bbox_inches='tight', transparent=True)
        plt.close(fig)
        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        seaborn.violinplot(data=df, x="Cluster", y=f)
        fig.savefig(f"./violin_{f}.png")
        fig.savefig(f"./violin_{f}.pdf", bbox_inches='tight', transparent=True)
        plt.close(fig)
    return df

def feature_to_rgb(minval, maxval, feature, cmap=plt.cm.YlGnBu):
    norm = matplotlib.colors.Normalize(vmin=minval, vmax=maxval)
    scalarmap = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
    rgb_feature = scalarmap.to_rgba(feature)
    return rgb_feature

def vec_to_rgb(df, feature_vec):
    rgb_vec = np.zeros(shape=(5,4))
    
    area_rgb = np.zeros(shape=(1,4))
    area_min = np.min(df.iloc[:, 0])
    area_max = np.max(df.iloc[:, 0])
    area_rgb[0] = feature_to_rgb(area_min, area_max, feature_vec[0])
    rgb_vec[0] = area_rgb
    
    intensity_rgb = np.zeros(shape=(1,4))
    intensity_min = np.min(df.iloc[:, 1])
    intensity_max = np.max(df.iloc[:, 1])
    intensity_rgb[0] = feature_to_rgb(intensity_min, intensity_max, feature_vec[1])
    rgb_vec[1] = intensity_rgb
    
    
    duration_rgb = np.zeros(shape=(1,4))
    duration_min = np.min(df.iloc[:, 2])
    duration_max = np.max(df.iloc[:, 2])
    duration_rgb[0] = feature_to_rgb(duration_min, duration_max, feature_vec[2])
    rgb_vec[2] = duration_rgb
    
    aspect_rgb = np.zeros(shape=(1,4))
    aspect_min = np.min(df.iloc[:, 3])
    aspect_max = np.max(df.iloc[:, 3])
    aspect_rgb[0] = feature_to_rgb(aspect_min, aspect_max, feature_vec[3])
    rgb_vec[3] = aspect_rgb
    
    solidity_rgb = np.zeros(shape=(1,4))
    solidity_min = np.min(df.iloc[:, 4])
    solidity_max = np.max(df.iloc[:, 4])
    solidity_rgb[0] = feature_to_rgb(solidity_min, solidity_max, feature_vec[4])
    rgb_vec[4] = solidity_rgb
    return rgb_vec
    
def compute_color_barcodes(df, kmeans, scaler):
    clusters = np.unique(kmeans.labels_)
    print("CLUSTERS")
    print(len(clusters))
     # maxima = []
    # for c in clusters:
    #     temp = np.mean(df.loc[df["Cluster"] == c], axis=0)
    #     print(f"Cluster {c}: {temp}")
    #     maxima.append(temp)
    maxima = reconstruct_points(kmeans.cluster_centers_, scaler)
    # df = reconstruct_points(df, scaler)
    rgba_arr = np.zeros(shape=(len(maxima), 5, 4))
    for i in range(len(maxima)):
        feature_vec = maxima[i]
        rgb_feature_vec = vec_to_rgb(df, feature_vec)
        rgba_arr[i] = rgb_feature_vec
    return rgba_arr

def plot_barcodes(df, kmeans, scaler):
    x_label_list = FEATURES[:-4]
    rgb_vec= compute_color_barcodes(df, kmeans, scaler) 
    print(rgb_vec.shape)
    rgb_vec = np.moveaxis(rgb_vec, 1, 0)
    print(rgb_vec.shape)
    fig = plt.figure()
    im = plt.imshow(rgb_vec, origin='lower', cmap=plt.cm.YlGnBu)
    plt.yticks(ticks=np.arange(len(x_label_list)),
               labels=x_label_list, fontsize=20)
    plt.tick_params(axis='x', left=False, labelleft=False)
    plt.colorbar(label="", orientation="vertical")
    fig.savefig('./mSCTs_barcode_flipped.png',
                bbox_inches='tight')
    fig.savefig('./mSCTs_barcode_flipped.pdf',
                transparent=True, bbox_inches='tight')
    plt.close()

def main():
    rprops_dict = collect_rprops()
    num_transients = compute_num_transients(rprops_dict)
    df = create_dataframe(rprops_dict, num_transients)
    df, scaler, clusterer, kmeans, movie_id, frames, ys, xs = kmeans_subtypes(df)
    df = compute_average_features_per_cluster(df, scaler, clusterer, kmeans, movie_id, frames, ys, xs)
    plot_barcodes(df, kmeans, scaler)

if __name__=="__main__":
    main()