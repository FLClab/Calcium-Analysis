import pickle
import torch
import glob
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
sys.path.insert(0, "..")
from train import MSCTDataset, LoadedEvent3D, preprocess_stream, UNet3D, CenterCrop
from torch.utils.data import Subset
import shutil
from tqdm import tqdm
import tifffile
import skimage.measure
import pandas
from metrics import CentroidDetectionError
import random
from ipywidgets import interact
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, required=True, help="Name of the model to load")
args = parser.parse_args()

result_folder = f"../../data/testset/UNet3D/{args.model}"

PROPERTIES=[
    "label", 
    "area", 
    "axis_major_length", 
    "axis_minor_length", 
    "bbox", 
    "centroid", 
    "intensity_max", 
    "intensity_min",
    "moments",
    "solidity"
]

prediction_model_folder = f"../../data/baselines/ZeroCostDL4Mic/{args.model}"
prediction_model_path = os.path.dirname(prediction_model_folder)
prediction_model_name = os.path.basename(prediction_model_folder)

Source_QC_folder = "../../data/testset/raw-input"
Target_QC_folder = "../../data/testset/manual-expert"

random_seed = 42
batch_size = 128
patch_size = 32

def quality_control_detection():

    quality_control_prediction = prediction_model_path+"/"+prediction_model_name+"/Quality Control Detection/Prediction"
    
    Source_QC_folder_tif = Source_QC_folder+"/*.tif"

    Z = sorted(glob.glob(Source_QC_folder_tif))
    target_Z = [file.replace(Source_QC_folder, Target_QC_folder) for file in Z]
    target_Z = []
    for file in Z:
        target_file = file.replace(Source_QC_folder, Target_QC_folder)
        dirname, basename = os.path.dirname(target_file), os.path.basename(target_file)
        name, ext = os.path.splitext(basename)
        target_Z.append(os.path.join(dirname, "".join(("manual_", name, ".csv"))))
    print("Number of test samples found in the folder: "+str(len(Z)))
    
    with open(os.path.join(prediction_model_path, prediction_model_name, "optimized-threshold-detection"), "r") as file:
        threshold = float(file.read())
    print(f"Loaded threshold from file: {threshold:0.3f}")
    
    out = {}
    for source_file, target_file in zip(tqdm(Z, desc="Images..."), target_Z):
        stream_raw = tifffile.imread(source_file)
        if stream_raw.ndim != 3:
            print("[!!!!] File does not appear to be a stream")
            print(f"[----] {source_file}")
            continue
        stream = preprocess_stream(stream_raw)
        
        stream_name = os.path.splitext(os.path.basename(source_file))[0] + "_prediction.tif"
        try:
            binary_prediction = tifffile.imread(os.path.join(quality_control_prediction, stream_name)) > 0
        except FileNotFoundError:
            continue

        # Label prediction
        label = skimage.measure.label(binary_prediction)
        regionprops_table = skimage.measure.regionprops_table(label, intensity_image=stream, properties=PROPERTIES)

        out[stream_name] = regionprops_table

    print("Done quality control detection!") 
    return out

def quality_control_segmentation():

    quality_control_prediction = prediction_model_path+"/"+prediction_model_name+"/Quality Control Segmentation/Prediction"
    
    Source_QC_folder_tif = Source_QC_folder+"/*.tif"

    Z = sorted(glob.glob(Source_QC_folder_tif))
    target_Z = [file.replace(Source_QC_folder, Target_QC_folder) for file in Z]
    target_Z = []
    for file in Z:
        target_file = file.replace(Source_QC_folder, Target_QC_folder)
        dirname, basename = os.path.dirname(target_file), os.path.basename(target_file)
        name, ext = os.path.splitext(basename)
        target_Z.append(os.path.join(dirname, "".join(("manual_", name, ".csv"))))
    print("Number of test samples found in the folder: "+str(len(Z)))
    
    with open(os.path.join(prediction_model_path, prediction_model_name, "optimized-threshold-segmentation"), "r") as file:
        threshold = float(file.read())
    print(f"Loaded threshold from file: {threshold:0.3f}")
    
    out = {}
    for source_file, target_file in zip(tqdm(Z, desc="Images..."), target_Z):
        stream_raw = tifffile.imread(source_file)
        if stream_raw.ndim != 3:
            print("[!!!!] File does not appear to be a stream")
            print(f"[----] {source_file}")
            continue
        stream = preprocess_stream(stream_raw)
        
        stream_name = os.path.splitext(os.path.basename(source_file))[0] + "_prediction.tif"
        try:
            binary_prediction = tifffile.imread(os.path.join(quality_control_prediction, stream_name)) > 0
        except FileNotFoundError:
            continue

        # Label prediction
        label = skimage.measure.label(binary_prediction)
        try:
            regionprops_table = skimage.measure.regionprops_table(label, intensity_image=stream, properties=PROPERTIES)
        except ValueError:
            print("Caught Value Error... Skipping file;")
            continue

        out[stream_name] = regionprops_table

    print("Done quality control segmentation!") 
    return out    

def main():
    # filename = os.path.join(prediction_model_path, prediction_model_name, "Quality Control Detection/QC_regionprops_"+prediction_model_name+".pkl")
    # regionprops = quality_control_detection()
    # with open(filename, "wb") as fp:
    #     pickle.dump(regionprops, fp)
    #     print("Detection dictionary saved successfully to pickle file")

    filename = os.path.join(prediction_model_path, prediction_model_name, "Quality Control Segmentation/QC_regionprops_"+prediction_model_name+".pkl")
    regionprops = quality_control_segmentation()
    with open(filename, "wb") as fp:
        pickle.dump(regionprops, fp)
        print("Segmentation dictionary saved successfully to pickle file")        

if __name__=="__main__":
    main()