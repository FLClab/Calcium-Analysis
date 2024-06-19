import numpy as np
import matplotlib.pyplot as plt 
import pandas
import pickle
import seaborn

model_keys = ["4-0", "1-0", "1-1", "1-2", "1-4", "1-8", "1-16", "1-32", "1-64", "1-128", "1-256"]
event_keys = ["onSynapse", "onDendrite", "smallArea", "bigArea", "outOfFocus", "longArea", "highIntensity", "lowIntensity"]

UNET_ORDER = np.array([1, 4, 7, 5, 6, 3, 2, 0])
UNET_KEYS = [event_keys[i] for i in UNET_ORDER]
STARDIST_ORDER = np.array([7, 4, 2, 1, 5, 3, 0, 6])
STARDIST_KEYS = [event_keys[i] for i in STARDIST_ORDER]

def load_data(model: str) -> np.ndarray:
    path = "./paper_figures/UNet_eventtype_theresa_segmentation.pkl" if model == "UNet" else "./paper_figures/StarDist_eventtype_theresa_segmentation.pkl"
    data = pickle.load(open(path, "rb"))
    data_array = np.zeros((len(model_keys), len(event_keys)))
    for i, mkey in enumerate(model_keys):
        for j, ekey in enumerate(event_keys):
            dice_list = data[ekey][mkey]
            dice_mean = np.mean(dice_list)
            data_array[i][j] = dice_mean
    return data_array

def compute_mean(data: np.ndarray, model: str) -> None:
    print(f"********** {model} *************")
    for k, j in zip(event_keys, range(data.shape[1])):
        col = data[:, j]
        meanval = np.mean(col)
        print(f"{k} mean = {meanval}")
    print("\n")


def plot_event_matrix(data: np.ndarray, model: str, order: np.ndarray, keys: list) -> None:
    data = data[:, order]
    fig = plt.figure()
    xticks = np.arange(0, len(event_keys), 1)
    yticks = np.arange(0, len(model_keys), 1)
    plt.imshow(data, cmap='RdPu', vmin=0.0, vmax=1.0)
    plt.xlabel('Event type')
    plt.ylabel('Models')
    plt.xticks(ticks=xticks, labels=keys, rotation=-45)
    plt.yticks(ticks=yticks, labels=model_keys)
    plt.colorbar(orientation='vertical')
    fig.savefig(f"./matrices/{model}_events.png")
    fig.savefig(f"./matrices/{model}_events.pdf", bbox_inches='tight', transparent=True)
    plt.close(fig)


def main():
    unet_data = load_data(model="UNet")
    stardist_data = load_data(model="StarDist")
    compute_mean(unet_data, "UNet")
    compute_mean(stardist_data, "StarDist")
    plot_event_matrix(data=unet_data, model='UNet', order=UNET_ORDER, keys=UNET_KEYS)
    plot_event_matrix(data=stardist_data, model='StarDist', order=STARDIST_ORDER, keys=STARDIST_KEYS)

if __name__=="__main__":
    main()