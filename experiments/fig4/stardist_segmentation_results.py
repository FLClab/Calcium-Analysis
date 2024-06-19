import pickle
import numpy
from matplotlib import pyplot 
import glob
import operator
from metrics import commons
from tqdm import tqdm
import argparse
import os
import scipy.stats
import tifffile
from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True, type=str, help="Directory containing the model's predictions")
parser.add_argument("--expert", required=False, type=str, default="mask_flavie_1")
args = parser.parse_args()

def save_dict(outdict, filename):
    with open(filename, "wb") as fp:
        pickle.dump(outdict, fp)
        print("Dictionary saved successfully to pickle file")

GROUND_TRUTH_PATH = "/home/frbea320/projects/def-flavielc/frbea320/mscts-analysis/data/testset"
PREDICTION_PATH = f"/home/frbea320/projects/def-flavielc/frbea320/mscts-analysis/data/testset/StarDist3D/{args.model}"

manual_expert_files = glob.glob(os.path.join(GROUND_TRUTH_PATH, "manual-expert", "*.csv"))


results = {
}

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

    start = tuple(map(lambda a, da: (numpy.round(a) - da //
                  2).astype(int), location, bounding))
    end = tuple(map(operator.add, start, bounding))
    if pad:
        padNumpy = []
        for low, up, arrShape in zip(start, end, arrayShape):
            pad_min = -low if low < 0 else 0
            pad_max = up - arrShape if (up - arrShape) > 0 else 0
            padNumpy.append((pad_min, pad_max))
        padNumpy = tuple(padNumpy)

    start = tuple(map(lambda a: numpy.clip(a, 0, None), start))
    end = tuple(map(lambda a, dmax: numpy.clip(
        a, 0, dmax).astype(int), end, array.shape))
    slices = tuple(map(slice, start, end))
    array = array[slices]

    if pad:
        array = numpy.pad(array, padNumpy, **kwargs)

    return array, slices


def baseline(y, lam=1e3, ratio=1e-6):
    """
    Provient de https://github.com/charlesll/rampy/blob/master/rampy/baseline.py
    """
    N = len(y)
    D = scipy.sparse.csc_matrix(numpy.diff(numpy.eye(N), 2))
    w = numpy.ones(N)
    MAX_ITER = 100

    for _ in range(MAX_ITER):
        W = scipy.sparse.spdiags(w, 0, N, N)
        Z = W + lam * D.dot(D.transpose())
        z = scipy.sparse.linalg.spsolve(Z, w * y)
        d = y - z
        # make d- and get w^t with m and s
        dn = d[d < 0]
        m = numpy.mean(dn)
        s = numpy.std(dn)
        wt = 1.0 / (1 + numpy.exp(2 * (d - (2 * s - m)) / s))
        # check exit condition and backup
        if numpy.linalg.norm(w - wt) / numpy.linalg.norm(w) < ratio:
            break
        w = wt

    return z


def get_deltaF(image, center):
    _, slices = get_crop(image, center, (1, 5, 5))
    trace = numpy.mean(image[:, slices[1], slices[2]], axis=(1, 2))

    # Compute normalized trace
    F0 = baseline(trace, lam=1e6, ratio=1e-10)

    trace_normalized = (trace - F0) / F0
    deltaF_frame = numpy.round(center[0]).astype(int)
    deltaF = trace_normalized[deltaF_frame]
    return deltaF

def get_segmentation_scores(data, tau):
    scores = {
        "dice": [],
        "is_empty": [],
    }
    for _, values in tqdm(data.items(), desc="Events types...", leave=False):
        for movie_id, coords, time_slice, index in tqdm(values, desc="Events..."):
            truth_name = glob.glob(os.path.join(GROUND_TRUTH_PATH, "segmentation-expert", args.expert, f"*-{index}.tif"))[0]
            truth = tifffile.imread(truth_name)
            x, y, width, height = coords
            pred_path = os.path.join(PREDICTION_PATH, movie_id.replace(".tif", ".npz"))
            pred_data = numpy.load(pred_path, allow_pickle=True)
            try:
                raw_pred = pred_data[tau]
            except:
                print(f"Prediction file {pred_path} is problematic")
                continue
            
            pred = raw_pred[time_slice, y:y+height, x:x+width]
            is_empty = numpy.count_nonzero(pred) == 0
            diceval = commons.dice(truth, pred)
            scores["dice"].append(diceval)
            scores["is_empty"].append(is_empty)
    return scores

def init_keys():
    file = manual_expert_files[0]
    filename = os.path.basename(file)
    prediction_file = filename.split("_")[-1].split(".")[0]
    prediction_file = os.path.join(PREDICTION_PATH, prediction_file + ".npz")
    pred_data = numpy.load(prediction_file)
    label_keys = pred_data.files
    return label_keys

def save_results(data, label_keys):
    model_dice = []
    model_scores = []
    for tau in tqdm(label_keys):
        scores = get_segmentation_scores(data, tau=tau)
        model_dice.append(numpy.mean(scores["dice"]))
        model_scores.append(scores)
    results["dice"] = model_dice
    results["all"] = model_scores
    results["thresholds"] = label_keys

def main():
    label_keys = init_keys()
    label_keys.remove('label')
    data = load_data()
    save_results(data, label_keys)
    print(results)
    save_dict(results, os.path.join(PREDICTION_PATH, "segmentation_results.pkl"))

if __name__=="__main__":
    main()