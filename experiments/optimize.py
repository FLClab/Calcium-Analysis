import numpy as np
import matplotlib.pyplot as plt
import os
from train import MSCTDataset, LoadedEvent3D, preprocess_stream, UNet3D
from torch.utils.data import Subset
import shutil
from tqdm import tqdm
import tifffile
import skimage.measure
import pandas
from metrics import CentroidDetectionError
import random
from ipywidgets import interact

source_qc_folder = "../data/calcium_dataset_crops.h5"
target_qc_folder = "../data/calcium_dataset_crops.h5"

QC_model_name = os.path.basename(QC_model_folder)
QC_model_path = os.path.dirname(QC_model_folder)
input_folder = "../data/testset/raw-input"
ground_truth_folder = "../data/testset/manual-expert"

random_seed = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
POSTPROCESS_PARAMS = {
    "minimal_time" : 2,
    "minimal_height" : 3,
    "minimal_width" : 3
}

def filter_regionprops(regionprops, prediction, constraints):
    updated_regionprops, remove_coords = [], []
    for rprop in regionprops:
        t1, h1, w1, t2, h2, w2 = rprop.bbox
        lenT = t2 - t1
        lenH = h2 - h1
        lenW = w2 - w1
        should_remove = False
        if "minimal_time" in constraints:
            if lenT < constraints["minimal_time"]:
                should_remove = True
            if lenH < constraints["minimal_height"]:
                should_remove = True
            if lenW < constraints["minimal_width"]:
                should_remove = True
        if should_remove:
            remove_coords.extend(rprop.coords)
        else:
            updated_regionprops.append(rprop)
    remove_coords = np.array(remove_coords)
    prediction[remove_coords[:, 0], remove_coords[:, 1], remove_coords[:, 2]] = 0    
    return updated_regionprops

def load_model():
    full_QC_model_path = QC_model_path+"/"+QC_model_name+"/"
    if os.path.exists(full_QC_model_path):
        print("The "+QC_model_name+" network will be evaluated")
        model = UNet3D.from_path(QC_model_path, QC_model_name)
        model = model.to(device)
    else:
         W  = '\033[0m'  # white (normal)
        R  = '\033[31m' # red
        print(R+'!! WARNING: The chosen model does not exist !!'+W)
        print('Please make sure you provide a valid model path and model name before proceeding further.')
    return model

def optimize_threshold(model, percentage=10):
    if os.path.isfile(source_qc_folder):
        valid_data = MSCTDataset(source_qc_folder, ["valid"])
    else:
        train_valid_data = MSCTDatasetFromFolder(source_qc_folder, target_qc_folder)
        np.random.seed(random_seed)
        indices = np.arange(len(train_valid_data))
        indices.shuffle()
        length = percentage * len(indices)
        valid_data = Subset(train_valid_data, indices[:length])
    valid_generator = LoadedEvent3D(valid_data, transforms=CenterCrop(patch_size))
    threshold = model.optimize_threshold(valid_generator)
    with open(os.path.join(qc_model_path, qc_model_name, "optimized_threshold"), "w") as file:
        file.write(str(threshold))
    print(f"The optimal threshold is: {threshold:0.3f}")
    
def quality_control(model):
    if os.path.exists(QC_model_path+"/"+QC_model_name+"/Quality Control/Prediction"):
        shutil.rmtree(QC_model_path+"/"+QC_model_name+"/Quality Control/Prediction")
    quality_control_prediction = QC_model_path+""+QC_model_name+"Quality Control/Prediction"
    os.makedirs(quality_control_prediction)
    source_QC_folder_tif = Source_QC_folder+"/*.tif"
    Z = sorted(glob.glob(Source_QC_folder_tif))
    target_Z = [file.replace(source_qc_folder, target_qc_folder) for file in Z]
    target_Z = []
    for file in Z:
        target_file = file.replace(source_qc_folder, target_qc_folder)
        dirname, basename = os.path.dirname(target_file), os.path.basename(target_file)
        name, ext = os.path.splitext(basename)
        target_Z.append(os.path.join(dirname, "".join(("manual_", name, ".csv"))))
    print("Number of test samples found in the folder: "+str(len(Z)))
    with open(os.path.join(QC_model_path, QC_model_name, "optimized-threshold"), "r") as file:
        threshold = float(file.read())
    print(f"Loaded threshold from file: {threshold:0.3f}")
    
    out = {}
    for source_file, target_file in zip(tqdm(Z, desc="Images..."), target_Z):
        stream_raw = tifffile.imread(source_file)
        if stream_raw.ndim != 3:
            print("[!!!!] File does not appear to be a stream")
            print(f"[----] {STREAM_FILE}")
            continue
        stream = preprocess_stream(stream_raw)
        prediction = model.predict_stream(
            {"input": stream, "shape2crop": np.array([patch_size * 2, patch_size * 2, patch_size * 2])},
            batch_size = batch_size,
            step = (patch_size, patch_size, patch_size),
            num_workers=4,
            device=DEVICE,
        )
        binary_prediction = (prediction > threshold).astype(int)
        label = skimage.measure.label(binary_prediction)
        regionprops = skimage.measure.regionprops(label, intensity_image=stream)
        updated_regionprops = filter_regionprops(regionprops, binary_prediction, POSTPROCESS_PARAMS)
        print("Calculating metrics...")
        
        out[source_file] = {
#         "seg-precision" : precision_score(truth.ravel(), binary_prediction.ravel()),
#         "seg-recall" : recall_score(truth.ravel(), binary_prediction.ravel()),
#         "seg-iou" : jaccard_score(truth.ravel(), binary_prediction.ravel()),
#         "seg-f1-score" : f1_score(truth.ravel(), binary_prediction.ravel()),        
        }    
        if os.path.isfile(target_file):
            truth_coords = pandas.read_csv(target_file)[["Slice", "X", "Y"]].to_numpy()
            pred_coords = [rprop.weighted_centroid for rprop in updated_regionprops]
            detector = CentroidDetectionError(truth_coords, pred_coords, algorithm="hungarian", threshold=6)
            out[source_file]["det-precision"] = detector.precision
            out[source_file]["det-recall"] = detector.recall
            out[source_fie]["det-f1-score"] = detector.f1_score
        else:
            out[source_file]["det-precision"] = -1
            out[source_file]["det-recall"] = -1
            out[source_file]["det-f1-score"] = 1
        print("Saving prediction...")
        savename = os.path.splitext(os.path.basename(source_file))[0] + "_prediction.tif"
        tifffile.imwrite(os.path.join(quality_control_prediction, savename), (binary_prediction * 255).astype(np.uint8))
        df = pandas.DataFrame.from_dict(out, orient="index")
        df.to_csv(os.path.join(QC_model_path, QC_model_name, "Quality Control/QC_metrics_"+QC_model_name+".csv"))
    
    print("Done predicting images!") 

def main():
    model = load_model()
    optimized_threshold(model)
    quality_control(model)

if __name__=="__main__":
    main()