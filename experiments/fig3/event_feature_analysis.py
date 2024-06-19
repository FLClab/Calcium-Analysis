import numpy as np 
import matplotlib.pyplot as plt 
import tifffile 
from tqdm import tqdm 
import os 
import glob 
import pandas 
from skimage import measure
from collections import defaultdict 
from metrics import CentroidDetectionError


GROUND_TRUTH_PATH = "../../data/testset"

MODELS_TO_EVAL = {
    "IDT": "/home/frbea320/projects/def-flavielc/frbea320/mscts-analysis/data/testset/minifinder",
    "StarDist_1-0": "/home/frbea320/projects/def-flavielc/frbea320/mscts-analysis/data/testset/StarDist3D/StarDist3D_subset-0.25-1_1-0_46",
    "UNet_1-0":  "/home/frbea320/projects/def-flavielc/frbea320/mscts-analysis/data/baselines/ZeroCostDL4Mic/unet3D-ZeroCostDL4Mic_subset-0.25-4_1-0_45",
    "UNet_1-64": "/home/frbea320/projects/def-flavielc/frbea320/mscts-analysis/data/baselines/ZeroCostDL4Mic/unet3D-ZeroCostDL4Mic_subset-0.25-1_1-64_46"
}

POSTPROCESS_PARAMS = {
            'minimal_time': 2,
            'minimal_height': 3,
            'minimal_width': 3,
        }

def normalize_movie(movie: np.ndarray):
    return (movie - np.min(movie)) / (np.max(movie) - np.min(movie))

manual_expert_files = glob.glob(os.path.join(GROUND_TRUTH_PATH, "manual-expert", "*.csv"))


def filter_rprops(regions, constraints):
    """
    Enleve les événements selon trop petit selon différents critères
    """
    regionRemovedCount = 0
    pbar = tqdm(regions, total=len(regions), leave=False)
    filtered_regions = []
    for region in pbar:
        area = region.area
        center = np.array(region.centroid)

        t1, h1, w1, t2, h2, w2 = region.bbox
        lenT = t2 - t1
        lenH = h2 - h1
        lenW = w2 - w1

        # Constraints check
        good = True
        if 'minimal_time' in constraints.keys():
            if lenT < constraints['minimal_time']:
                good = False
        if 'minimal_height' in constraints.keys():
            if lenH < constraints['minimal_height']:
                good = False
        if 'minimal_width' in constraints.keys():
            if lenW < constraints['minimal_width']:
                good = False
        if 'maximal_time' in constraints.keys():
            if lenT > constraints['maximal_time']:
                good = False
        if good:
            filtered_regions.append(region)
        else:
            regionRemovedCount += 1
        pbar.set_description(f'{regionRemovedCount} Regions removed so far')
    return filtered_regions


def get_event_features(model_path: str, model_name: str):
    skipped_files = []
    intensity_data = []
    for file in tqdm(manual_expert_files, desc="Videos..."):
        data = pandas.read_csv(file)
        truth_centroids = np.stack((data["Slice"], data["Y"], data["X"]), axis=-1)
        filename = os.path.basename(file)
        movie_id = filename.split("_")[-1]
        moviefile = f"{GROUND_TRUTH_PATH}/raw-input/{movie_id}".replace(".csv", ".tif")
        movie = tifffile.imread(moviefile)
        movie = normalize_movie(movie)
        try: 
            if "unet" in model_name.lower():
                pred = tifffile.imread(f"{model_path}/Quality Control Segmentation/Prediction/{movie_id}".replace(".csv", "_prediction.tif"))
            elif "idt" in model_name.lower():
                pred = np.load(f"{model_path}/{movie_id}".replace(".csv", ".npy"))
            else:
                pred = np.load(f"{model_path}/{movie_id}".replace(".csv", ".npz"))['label']
        except FileNotFoundError:
            print(f"{filename} not found")
            skipped_files.append(filename)
            continue
        pred_label = measure.label(pred)
        pred_rprops = measure.regionprops(label_image=pred_label, intensity_image=movie)
        pred_rprops = filter_rprops(regions=pred_rprops, constraints=POSTPROCESS_PARAMS)
        pred_centroids = np.array([r.weighted_centroid for r in pred_rprops])
        detector = CentroidDetectionError(truth_centroids, pred_centroids, threshold=6, algorithm='hungarian')
        truth_couple, pred_couple = detector.get_coupled()
        fp = detector.get_false_positives()
        fn = detector.get_false_negatives()
        temp = []
        for idx in pred_couple:
            region = pred_rprops[idx]
            temp.append((region.max_intensity, 0))
        for idx in fp:
            region = pred_rprops[idx]
            temp.append((region.max_intensity, 1))
        for idx in fn:
            t, y, x = truth_centroids[idx]
            t, y, x = int(t), int(y), int(x)
            temp.append((movie[t, y, x], 2))


        intensity_data.extend(temp)

    np.save(f"./intensity_data/{model_name}.npy", np.array(intensity_data))
    print(f"{model_path} skipped files: {skipped_files}")
           

def main():
    for key in MODELS_TO_EVAL.keys():
        get_event_features(model_path=MODELS_TO_EVAL[key], model_name=key)

if __name__=="__main__":
    main()