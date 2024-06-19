import numpy as np 
import matplotlib.pyplot as plt
import pickle 
import tifffile 
from tqdm import tqdm 
import os 
import glob 
import pandas 
from scipy.stats import wasserstein_distance
import scipy
from skimage import measure
from collections import defaultdict
import pandas
import ot
from wasserstein_distance_nd import wasserstein_distance_nd


DATAPATH = "/home/frbea320/projects/def-flavielc/frbea320/mscts-analysis/data/testset"
GROUND_TRUTH = "/home/frbea320/projects/def-flavielc/frbea320/mscts-analysis/data/testset/segmentation-expert/mask_theresa_1"

MODELS_TO_EVAL = {
    "StarDist_4-0": "/home/frbea320/projects/def-flavielc/frbea320/mscts-analysis/data/testset/StarDist3D/StarDist3D_complete_1-0_46",
    "StarDist_1-0": "/home/frbea320/projects/def-flavielc/frbea320/mscts-analysis/data/testset/StarDist3D/StarDist3D_subset-0.25-1_1-0_46",
    "UNet_4-0": "/home/frbea320/projects/def-flavielc/frbea320/mscts-analysis/data/baselines/ZeroCostDL4Mic/unet3D-ZeroCostDL4Mic_complete_1-0_44",
    "UNet_1-0":  "/home/frbea320/projects/def-flavielc/frbea320/mscts-analysis/data/baselines/ZeroCostDL4Mic/unet3D-ZeroCostDL4Mic_subset-0.25-4_1-0_45",
    "UNet_1-64": "/home/frbea320/projects/def-flavielc/frbea320/mscts-analysis/data/baselines/ZeroCostDL4Mic/unet3D-ZeroCostDL4Mic_subset-0.25-1_1-64_46"
}

POSTPROCESS_PARAMS = {
            'minimal_time': 2,
            'minimal_height': 3,
            'minimal_width': 3,
        }

def normalize_crop(crop):
    return (crop - np.min(crop)) / (np.max(crop) - np.min(crop))


def filter_rprops(regions: list, constraints: dict = POSTPROCESS_PARAMS):
    """
    Enleve les événements selon trop petit selon différents critères
    """

    filtered_regions = []
    for region in regions:
        area = region.area
        center = np.array(region.centroid)

        h1, w1, h2, w2 = region.bbox
        lenH = h2 - h1
        lenW = w2 - w1

        # Constraints check
        good = True
        if 'minimal_height' in constraints.keys():
            if lenH < constraints['minimal_height']:
                good = False
        if 'minimal_width' in constraints.keys():
            if lenW < constraints['minimal_width']:
                good = False
        if good:
            filtered_regions.append(region)
    return filtered_regions

def load_data():
    with open("../../baselines/files/macro.ijm", "r") as file:
        lines = [
            line.rstrip() for line in file.readlines()
            if (line.startswith("eventType")) or
               (line.startswith("open(rawFolder")) or
               (line.startswith("makeRectangle(")) or
               (line.startswith("duplicateAndSave("))
        ]

    out = defaultdict(list)
    key = None
    i = 1
    flag = False
    for line in lines:
        if line.startswith("eventType"):
            key = line.split('"')[-2]
            continue
        if line.startswith("open("):
            line = line.split('"')[-2]
        if line.startswith("makeRectangle("):
            line = eval(line.split("makeRectangle")[-1][:-1])
        if isinstance(line, str) and line.startswith("duplicateAndSave("):
            flag = True
            line = eval(line.split("duplicateAndSave(")[-1].split(",")[0])
        out[key].append(line)
        if flag:
            out[key].append(i)
            i += 1
            flag = False
    for key, values in out.items():
        new_values = []
        tmp = []
        for i, value in enumerate(values):
            if (i > 0) and (i % 4 == 0):
                new_values.append(tmp)
                tmp = []
            tmp.append(value)
        out[key] = new_values
    return out

def compare_feature_distributions(data: dict, model_path: str, label: str):
    all_gt_features = []
    all_model_features = []
    model_name = model_path.split("/")[-1]
    feature_names = ["area", "aspect ratio", "extent", "mean intensity", "max intensity", "min intensity", "orientation", "perimeter", "solidity"]
    num_skipped = []
    for event_type, values in tqdm(data.items(), desc="Event types...", leave=False):
        for movie_id, coords, time_slice, index in tqdm(values, desc="Events..."):
            movie = tifffile.imread(os.path.join(DATAPATH, "raw-input", movie_id))
            truth_name = glob.glob(os.path.join(
                DATAPATH, "segmentation-expert", "mask_theresa_1", f"*-{index}.tif"))[0]
            truth = tifffile.imread(truth_name)
            x, y, width, height = coords 
            movie_crop = normalize_crop(movie[time_slice, y:y+height, x:x+width])
            try:
                if "unet3d" in model_name.lower():
                    full_pred = tifffile.imread(f"{model_path}/Quality Control Segmentation/Prediction/{movie_id}".replace(".tif", "_prediction.tif"))
                    pred = full_pred[time_slice, y:y+height, x:x+width]
                else:
                    full_pred = np.load(f"{model_path}/{movie_id}".replace(".tif", ".npz"))['label']
                    pred = full_pred[time_slice, y:y+height, x:x+width]
            except FileNotFoundError:
                num_skipped.append(movie_id)
                continue
            
            # TODO: compute rprops for both model and ground truth
            gt_label = measure.label(truth)
            gt_regions = measure.regionprops(label_image=gt_label, intensity_image=movie_crop)
            gt_regions = filter_rprops(gt_regions)
            pred_label = measure.label(pred)
            pred_regions = measure.regionprops(label_image=pred_label, intensity_image=movie_crop)
            pred_regions = filter_rprops(pred_regions)
            for r in gt_regions:
                aspect_ratio = r.major_axis_length / r.minor_axis_length 
                data_gt = np.array([r.area, aspect_ratio, r.extent, r.mean_intensity, r.max_intensity, r.min_intensity, r.orientation, r.perimeter, r.solidity])
                all_gt_features.append(data_gt)

            for r in pred_regions:
                aspect_ratio = r.major_axis_length / r.minor_axis_length
                data_pred = np.array([r.area, aspect_ratio, r.extent,r.mean_intensity, r.max_intensity, r.min_intensity, r.orientation, r.perimeter, r.solidity])
                all_model_features.append(data_pred)

    num_skipped = list(set(num_skipped))
    print(f"* Skipped {len(num_skipped)} files *")
    gt_features = np.r_[all_gt_features]
    model_features = np.r_[all_model_features]
    wasser = wasserstein_distance_nd(u_values=gt_features, v_values=model_features)
    print(f"Full distribution wasserstein = {wasser}")
    print(gt_features.shape, model_features.shape)
    compute_1d_distance(features1=gt_features, features2=model_features, feature_names=feature_names, model_name=model_name)
    gt_df = pandas.DataFrame(data=gt_features, columns=feature_names)
    gt_df["model"] = ["Ground truth"] * gt_df.shape[0]
    pred_df = pandas.DataFrame(data=model_features, columns=feature_names)
    pred_df["model"] = [label] * pred_df.shape[0]
    return gt_df, pred_df
    # TODO compute nd wasserstein distance


def compute_1d_distance(features1: np.ndarray, features2: np.ndarray, feature_names: list, model_name: str) -> None:
    print(features1.shape, features2.shape)
    assert features1.shape[1] == features2.shape[1]
    features_id = np.arange(0, features1.shape[1], 1)
    print(f"--- {model_name} ---")
    for name, f in zip(feature_names, features_id):
        u_values, v_values = features1[:, f], features2[:, f]
        u_mean, v_mean = np.mean(u_values), np.mean(v_values)
        feature_diff = u_mean - v_mean

        distance = wasserstein_distance(u_values=u_values, v_values=v_values)
        print(f"\t{name} Wasserstein: {distance}\t\tDifference: {feature_diff}")
    print("--- ... ---\n")

def compute_nd_distance():
    pass

def main():
    pass

if __name__=="__main__":
    main()