import numpy as np
import matplotlib.pyplot as plt
import pickle
import tifffile
from tqdm import tqdm
import glob
import pandas
import os
import operator
import skimage.measure
from tp_fp_fn import baseline, get_crop, get_deltaF, get_movie, compute_cost_matrix, assign_hungarian, get_fp, get_fn, get_precision, get_recall, get_f1
from SD_tpfpfn import filter_rprops

RESULTS_PATH = "/home/frbea320/projects/def-flavielc/anbil106/MSCTS-Analysis/testset/UNet3D/"
GROUND_TRUTH_PATH = "/home/frbea320/projects/def-flavielc/frbea320/mscts-analysis/data/testset"
RAW_PATH = "/home/frbea320/projects/def-flavielc/frbea320/mscts-analysis/data/testset/raw-input"
BASE_PATH = f"/home/frbea320/projects/def-flavielc/frbea320/mscts-analysis/data/baselines/ZeroCostDL4Mic"
manual_expert_files = glob.glob(os.path.join(GROUND_TRUTH_PATH, "manual-expert", "*.csv"))

POSTPROCESS_PARAMS = {
            'minimal_time': 3,
            'minimal_height': 3,
            'minimal_width': 3,
        }

def compute_histogram(deltaF_tp, deltaF_fp, deltaF_fn):
    max_tp = int(np.ceil(max(deltaF_tp)))
    max_fp = int(np.ceil(max(deltaF_fp)))
    max_fn = int(np.ceil(max(deltaF_fn)))
    bins_for_all = np.arange(0, 7, 0.5)
    tp_hist, _ = np.histogram(deltaF_tp, bins_for_all)
    fp_hist, _ = np.histogram(deltaF_fp, bins_for_all)
    fn_hist, _ = np.histogram(deltaF_fn, bins_for_all)
    return tp_hist, fp_hist, fn_hist

def evaluate_IDT():
    all_tp, all_fp, all_fn = 0, 0, 0
    deltaF_tp, deltaF_fp, deltaF_fn = [], [], []
    for f in tqdm(manual_expert_files, desc="Manual annotations..."):
        fname = f.split("/")[-1].split("_")[1].split(".")[0]
        movie = get_movie(fname)
        data = pandas.read_csv(f)
        truth_centroids = np.stack((data["Slice"], data["Y"], data["X"]), axis=-1)
        minifinder_name = os.path.join(GROUND_TRUTH_PATH, "minifinder", f"{fname}.npy")
        minifinder_pred = np.load(minifinder_name)
        minifinder_label = skimage.measure.label(minifinder_pred)
        minifinder_rprops = skimage.measure.regionprops(minifinder_label, intensity_image=movie)
        minifinder_rprops = filter_rprops(regions=minifinder_rprops, constraints=POSTPROCESS_PARAMS)
        minifinder_centroids = [r.weighted_centroid for r in minifinder_rprops]
        truth_deltaF = [get_deltaF(movie, c) for c in truth_centroids]
        pred_deltaF = [get_deltaF(movie, c) for c in minifinder_centroids]
        cost_matrix = compute_cost_matrix(truth_centroids, minifinder_centroids)
        truth_couple, pred_couple = assign_hungarian(cost_matrix)
        fn = get_fn(cost_matrix, truth_couple)
        fp = get_fp(cost_matrix, pred_couple)
        all_tp += pred_couple.shape[0]
        all_fp += fp.shape[0]
        all_fn += fn.shape[0]
        F_tp = [pred_deltaF[i] for i in pred_couple.tolist()]
        F_fp = [pred_deltaF[i] for i in fp.tolist()]
        F_fn = [truth_deltaF[i] for i in fn.tolist()]
        deltaF_tp += F_tp
        deltaF_fp += F_fp
        deltaF_fn += F_fn
    tp_data, fp_data, fn_data = compute_histogram(deltaF_tp, deltaF_fp, deltaF_fn)
    return tp_data, fp_data, fn_data

def main():
    tp_array, fp_array, fn_array = evaluate_IDT()
    np.savez("./IDT_tp_fp_fn.npz", tp_array=tp_array, fp_array=fp_array, fn_array=fn_array)

if __name__=="__main__":
    main()


