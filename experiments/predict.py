import numpy as np
import matplotlib.pyplot as plt
import tifffile
import torch
import os
import glob
from tqdm import tqdm
from train import UNet3D, preprocess_stream
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, required=True, help="Name of the model to load")
parser.add_argument("--data-folder", type=str, default="../data/testset/raw-input", help="Name of the model to load")
parser.add_argument("--result-folder", type=str, default="../data/testset/UNet3D", help="Name of the model to load")
args = parser.parse_args()

data_folder = args.data_folder
result_folder = os.path.join(args.result_folder, args.model)

prediction_model_folder = f"../data/baselines/ZeroCostDL4Mic/{args.model}"
prediction_model_path = os.path.dirname(prediction_model_folder)
prediction_model_name = os.path.basename(prediction_model_folder)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 64
patch_size = 32

def load_model():
    full_prediction_model_path = prediction_model_path+"/"+prediction_model_name+"/"
    if os.path.exists(full_prediction_model_path):
        print("The "+prediction_model_name+" network will be used.")
    else:
        W  = '\033[0m'  # white (normal)
        R  = '\033[31m' # red
        print(R+'!! WARNING: The chosen model does not exist !!'+W)
        print('Please make sure you provide a valid model path and model name before proceeding further.')
    os.makedirs(result_folder, exist_ok=True)

    model = UNet3D.from_path(prediction_model_path, prediction_model_name)
    model = model.to(DEVICE)
    
    return model

def generate_predictions(model):
    source_qc_folder_tif = os.path.join(data_folder, "**/*.tif")
    Z = sorted(glob.glob(source_qc_folder_tif, recursive=True))
    print("Number of test samples in the folder: "+str(len(Z)))
    for source_file in tqdm(Z, desc="Images..."):        
        stream_raw = tifffile.imread(source_file)
        if stream_raw.ndim != 3:
            print("[!!!!] File does not appear to be a stream")
            print(f"[----] {STREAM_FILE}")
            continue
        stream = preprocess_stream(stream_raw)
        prediction = model.predict_stream(
            {"input": stream, "shape2crop": np.array([patch_size * 2, patch_size*2, patch_size*2])},
            batch_size = batch_size * 2,
            step = (patch_size, patch_size, patch_size),
            num_workers=4,
            device=DEVICE,
        )
        
        experiment_name = os.path.basename(data_folder)
        savename = source_file.split(data_folder)[-1]
        if savename.startswith(os.path.sep):
            savename = savename[1:]
        dirname = os.path.dirname(savename)
        savename = os.path.splitext(os.path.basename(source_file))[0] + "_prediction.tif"        
        
        savename = os.path.join(result_folder, experiment_name, dirname, savename)
        dirname = os.path.dirname(savename)
        os.makedirs(dirname, exist_ok=True)
        tifffile.imwrite(savename, prediction.astype(np.float32))
    print("Images saved into folder: ", result_folder)

def main():
    model = load_model()
    generate_predictions(model)
    
if __name__=="__main__":
    main()

