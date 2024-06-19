import numpy as np
import matplotlib.pyplot as plt
import glob
from model_paths import UNet_025
import tifffile
from skimage.measure import label, regionprops
import pickle
from matplotlib import colors
from tqdm import tqdm
from collections import defaultdict
import operator
from metrics import CentroidDetectionError
from metrics.segmentation import commons

gray_cm = plt.get_cmap('gray', 256)
green_colors = gray_cm(np.linspace(0, 1, 256))
green_colors[:, [0,2]] = 0.0
green_cm = colors.ListedColormap(green_colors)

BASE_PATH = "/home/frbea320/projects/def-flavielc/frbea320/mscts-analysis/data/baselines/ZeroCostDL4Mic"
RAW_PATH = "/home/frbea320/projects/def-flavielc/frbea320/mscts-analysis/data/testset/raw-input"
EXPERT_PATH = "/home/frbea320/projects/def-flavielc/frbea320/mscts-analysis/data/testset/segmentation-expert/mask_flavie_1"
DATA_PATH = "/home/frbea320/projects/def-flavielc/frbea320/mscts-analysis/baselines/files"


best_1_0 = 2
best_25_0 = 23
best_25_1 = 17
best_25_2 = 23
best_25_4 = 23
best_25_8 = 21


def baseline(y, lam=1e3, ratio=1e-6):
    """
    Provient de https://github.com/charlesll/rampy/blob/master/rampy/baseline.py
    """
    N = len(y)
    D = scipy.sparse.csc_matrix(np.diff(np.eye(N), 2))
    w = np.ones(N)
    MAX_ITER = 100

    for _ in range(MAX_ITER):
        W = scipy.sparse.spdiags(w, 0, N, N)
        Z = W + lam * D.dot(D.transpose())
        z = scipy.sparse.linalg.spsolve(Z, w * y)
        d = y - z
        # make d- and get w^t with m and s
        dn = d[d < 0]
        m = np.mean(dn)
        s = np.std(dn)
        wt = 1.0 / (1 + np.exp(2 * (d - (2 * s - m)) / s))
        # check exit condition and backup
        if np.linalg.norm(w - wt) / np.linalg.norm(w) < ratio:
            break
        w = wt

    return z

def load_data():
    with open(f"{DATA_PATH}/macro.ijm", "r") as file:
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

def get_crop(array, location, bounding, pad=False, **kwargs):
    """
    Permet de retourner les coordonnés du pour faire le crop dans l'array donnée.
    Si les dimensions du crop dépassent celles de l'array, le crop sera plus petit.
    INPUT:
        array: array quelconque
        location: spécifie le centre du crop
        size: si un seul chiffre, ce chiffre est utilisé comme longueur pour chaque dimension,
              sinon size doit avoir la même dimension que array
        pad: si pad est True, on pad le array pour avoir la bonne size
    OUPUT:
        retourne les coordonnées pour effectuer le crop dans l'array
    """
    arrayShape = array.shape

    if len(bounding) != len(arrayShape):
        ValueError("The size of the crop should be the same size as the array")

    start = tuple(map(lambda a, da: (np.round(a) - da // 2).astype(int), location, bounding))
    end = tuple(map(operator.add, start, bounding))
    if pad:
        padNumpy = []
        for low, up, arrShape in zip(start, end, arrayShape):
            pad_min = -low if low < 0 else 0
            pad_max = up - arrShape if (up - arrShape) > 0 else 0
            padNumpy.append((pad_min, pad_max))
        padNumpy = tuple(padNumpy)

    start = tuple(map(lambda a: np.clip(a, 0, None), start))
    end = tuple(map(lambda a, dmax: np.clip(a, 0, dmax).astype(int), end, array.shape))
    slices = tuple(map(slice, start, end))
    array = array[slices]

    if pad:
        array = np.pad(array, padNumpy, **kwargs)

    return array, slices
    
def get_deltaF(image, center):
    _, slices = get_crop(image, center, (1, 5, 5))
    trace = np.mean(image[:, slices[1], slices[2]], axis=(1, 2))

    # Compute normalized trace
    F0 = baseline(trace, lam=1e6, ratio=1e-10)

    trace_normalized = (trace - F0) / F0
    deltaF_frame = np.round(center[0]).astype(int)
    deltaF = trace_normalized[deltaF_frame]
    return deltaF

def is_empty(prediction):
    num_nonzero = np.count_nonzero(prediction)
    return num_nonzero == 0

def get_prediction_files(model, task="Detection"):
    path = f"{BASE_PATH}/{model}/Quality Control {task}/Prediction"
    rprops_dict = f"{BASE_PATH}/{model}/Quality Control {task}/QC_regionprops_{model}.pkl"
    rprops = pickle.load(open(rprops_dict, "rb"))
    files = glob.glob(f"{path}/*")
    return files, rprops

def get_specific_moviemodel(pu_config, best_seed, movie, centroid):
    model = UNet_025[pu_config][best_seed]
    pred_detection = f"{BASE_PATH}/{model}/Quality Control Detection/Prediction/{movie}"
    pred_segmentation = f"{BASE_PATH}/{model}/Quality Control Segmentation/Prediction/{movie}"
    pred_detection = tifffile.imread(pred_detection)
    pred_segmentation = tifffile.imread(pred_segmentation)
    t, y, x = centroid[0], centroid[1], centroid[2]
    pred_detection_crop = pred_detection[t, y-32:y+32, x-32:x+32]
    pred_segmentation_crop = pred_segmentation[t, y-32:y+32, x-32:x+32]
    return pred_detection_crop, pred_segmentation_crop

def get_all_models_predictions(models, movie_id, coords, time_slice):
    detection_predictions = []
    segmentation_predictions = []
    found_empty = False
    x, y, width, height = coords
    for model in models:
        detection_path = f"{BASE_PATH}/{model}/Quality Control Detection/Prediction/{movie_id}".replace(".tif", "_prediction.tif")
        segmentation_path = f"{BASE_PATH}/{model}/Quality Control Segmentation/Prediction/{movie_id}".replace(".tif", "_prediction.tif")
        detection_pred = tifffile.imread(detection_path)
        segmentation_pred = tifffile.imread(segmentation_path)
        detection_crop = detection_pred[time_slice, y:y+height, x:x+width]
        segmentation_crop = segmentation_pred[time_slice, y:y+height, x:x+width]
        detection_predictions.append(detection_crop)
        segmentation_predictions.append(segmentation_crop)
        if not found_empty:
            found_empty = is_empty(detection_crop) or is_empty(segmentation_crop)
    return detection_predictions, segmentation_predictions, found_empty

def plot_predictions(detection_predictions, segmentation_predictions, truth, movie_crop, index):
    labels = ["4-0", "1-0", "1-1", "1-2", "1-4", "1-8"]
    assert len(detection_predictions) == len(segmentation_predictions)
    assert len(detection_predictions) == len(labels)
    fig, axs = plt.subplots(len(detection_predictions), 4)
    for i, (detection_model_pred, segmentation_model_pred) in enumerate(zip(detection_predictions, segmentation_predictions)):
        axs[i][0].imshow(movie_crop, cmap=green_cm)
        axs[i][1].imshow(truth, cmap='gray')
        axs[i][2].imshow(detection_model_pred, cmap='gray')
        axs[i][3].imshow(segmentation_model_pred, cmap='gray')
    axs[0][0].set_title("Crop")
    axs[0][1].set_title("Ground truth")
    axs[0][2].set_title("Detection\noptimized")
    axs[0][3].set_title("Segmentation\noptimized")
    for ax in axs.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
    fig.savefig("./UNet_example_masks/temp_{}.png".format(index), bbox_inches='tight')
    plt.close(fig)    

def get_corresponding_movie(prediction_file):
    fname = prediction_file.split("/")[-1].replace("_prediction", "")
    movie = tifffile.imread(f"{RAW_PATH}/{fname}")
    return movie

def find_empty_predictions(data, models):
    # TODO: COMPUTE PERCENTAGE OF MISSED TRANSIENTS
    index = 0
    for _, values in tqdm(data.items(), desc="Event types...", leave=False):
        for movie_id, coords, time_slice, index in tqdm(values, desc="Events..."):
            movie = tifffile.imread(f"{RAW_PATH}/{movie_id}")
            truth_name = glob.glob(f"{EXPERT_PATH}/*-{index}.tif")[0]
            truth = tifffile.imread(truth_name)
            x, y, width, height = coords
            movie_crop = movie[time_slice, y:y+height, x:x+width]
            detection_predictions, segmentation_predictions, found_empty = get_all_models_predictions(models, movie_id, coords, time_slice)
            if found_empty:
                plot_predictions(detection_predictions, segmentation_predictions, truth, movie_crop, index)
                index += 1
            else:
                continue
            

def get_crops_from_prediction(files, rprops):
    f = np.random.choice(files, size=1)[0]
    pred_id = f.split("/")[-1]
    pred = tifffile.imread(f)
    segmentation_pred = tifffile.imread(f.replace("Quality Control Detection", "Quality Control Segmentation"))
    movie = get_corresponding_movie(f)
    pred_rprops = rprops[pred_id]
    num_minis = len(pred_rprops["centroid-0"])
    mini_ids = np.random.choice(list(range(num_minis)), size=10)
    for i in mini_ids:
        t, y, x = int(pred_rprops["centroid-0"][i]), int(pred_rprops["centroid-1"][i]), int(pred_rprops["centroid-2"][i])
        movie_crop = movie[t, y-32:y+32, x-32:x+32]
        detection_pred_crop = pred[t, y-32:y+32, x-32:x+32]
        segmentation_pred_crop = segmentation_pred[t, y-32:y+32, x-32:x+32]
        detection_pred_crop_1_0, segmentation_pred_crop_1_0 = get_specific_moviemodel("UNet3D_complete_1-0", best_1_0, pred_id, centroid=(t, y, x))
        detection_pred_crop_25_1, segmentation_pred_crop_25_1 = get_specific_moviemodel("25-1", best_25_1, pred_id, centroid=(t, y, x))
        detection_pred_crop_25_2, segmentation_pred_crop_25_2 = get_specific_moviemodel("25-2", best_25_2, pred_id, centroid=(t, y, x))
        detection_pred_crop_25_4, segmentation_pred_crop_25_4 = get_specific_moviemodel("25-4", best_25_4, pred_id, centroid=(t, y, x))
        detection_pred_crop_25_8, segmentation_pred_crop_25_8 = get_specific_moviemodel("25-8", best_25_8, pred_id, centroid=(t, y, x))
        
        # fig, axs = plt.subplots(1, 7)
        # axs[0].imshow(movie_crop, cmap=green_cm)
        # axs[1].imshow(pred_crop_1_0, cmap='gray')
        # axs[2].imshow(pred_crop, cmap='gray')
        # axs[3].imshow(pred_crop_25_1, cmap='gray')
        # axs[4].imshow(pred_crop_25_2, cmap='gray')
        # axs[5].imshow(pred_crop_25_4, cmap='gray')
        # axs[6].imshow(pred_crop_25_8, cmap='gray')
        # for ax in axs:
        #     ax.set_xticks([])
        #     ax.set_yticks([])
        # fig.savefig("./UNet_example_masks/temp{}.png".format(i))
        # plt.close(fig)
    

def main():
    files, rprops = get_prediction_files(UNet_025["25-0"][best_25_0])
    data = load_data()
    models = [
        UNet_025["UNet3D_complete_1-0"][best_1_0],
        UNet_025["25-0"][best_25_0],
        UNet_025["25-1"][best_25_1],
        UNet_025["25-2"][best_25_2],
        UNet_025["25-4"][best_25_4],
        UNet_025["25-8"][best_25_8]
    ]
    find_empty_predictions(
        data=data,
        models=models
    )
    

if __name__=="__main__":
    main()