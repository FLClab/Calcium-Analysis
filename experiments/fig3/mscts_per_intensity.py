import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import scipy.sparse.linalg
import operator
from tqdm import tqdm
import pandas
import tifffile
from typing import Tuple

GROUND_TRUTH_PATH = "/home/frbea320/projects/def-flavielc/frbea320/mscts-analysis/data/testset"
RAW_PATH = "/home/frbea320/projects/def-flavielc/frbea320/mscts-analysis/data/testset/raw-input"


manual_expert_files = glob.glob(os.path.join(GROUND_TRUTH_PATH, "manual-expert", "*.csv"))

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

def get_deltaF(image: np.ndarray, center: Tuple) -> float:
    _, slices = get_crop(image, center, (1, 5, 5))
    trace = np.mean(image[:, slices[1], slices[2]], axis=(1, 2))

    # Compute normalized trace
    F0 = baseline(trace, lam=1e6, ratio=1e-10)

    trace_normalized = (trace - F0) / F0
    deltaF_frame = np.round(center[0]).astype(int)
    deltaF = trace_normalized[deltaF_frame]
    return deltaF

def get_movie(fname: str) -> np.ndarray:
    movie = tifffile.imread(f"{RAW_PATH}/{fname}.tif")
    return movie


def get_mscts() -> list:
    all_intensities = []
    for f in tqdm(manual_expert_files, desc="Manual annotations..."):
        fname = f.split("/")[-1].split("_")[1].split(".")[0]
        movie = get_movie(fname)
        data = pandas.read_csv(f)
        truth_centroids = np.stack((data["Slice"], data["Y"], data["X"]), axis=-1)
        intensities = [get_deltaF(movie, c) for c in truth_centroids]
        all_intensities += intensities
    return all_intensities

def compute_bins(intensities: list) -> None:
    max_x = int(np.ceil(max(intensities)))
    print(max_x)
    x = np.arange(0, max_x, 0.5)
    xlabels = [str(item) for item in x]
    fig = plt.figure()
    plt.hist(intensities, bins=x, edgecolor='black', color='tab:purple', log=True)
    plt.xlabel('Intensity')
    plt.ylabel('# mSCTs')
    plt.xticks(ticks=x, labels=xlabels)
    fig.savefig("./figures/TPFPFN_matrices/all_msct_inensities.png")
    fig.savefig("./figures/TPFPFN_matrices/all_msct_inensities.pdf", transparent=True, bbox_inches='tight')

def main():
    intensities = get_mscts()
    compute_bins(intensities=intensities)

if __name__=="__main__":
    main()