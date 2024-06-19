import numpy as np
import matplotlib.pyplot as plt
import argparse
from typing import Tuple

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="UNet")
args = parser.parse_args()


CLIP_VALUE = 150

def plot_fancy_histogram(data, expr, rate: bool = False):
    fig = plt.figure()
    plt.imshow(data, cmap='RdPu', vmin=0.0, vmax=1.0)
    plt.xlabel('Delta F / F')
    plt.ylabel('Models')
    # plt.xticks(ticks=xticks, labels=xlabels)
    # plt.yticks(ticks=yticks, labels=ylabels)
    plt.colorbar(orientation="vertical") 
    ratestr = "rate" if rate else ""
    fig.savefig(f"./figures/TPFPFN_matrices/{expr}_histogram_{ratestr}.png", bbox_inches='tight')
    fig.savefig(f"./figures/TPFPFN_matrices/{expr}_histogram_{ratestr}.pdf", transparent=True, bbox_inches='tight')
    plt.close(fig)

def plot_recall_matrix(recall: np.ndarray) -> None:
    fig = plt.figure()
    x = np.arange(0, recall.shape[1], 1)
    y = np.arange(0, recall.shape[0], 1)
    xlabels = ['0.0', '0.5', '1.0', '1.5', '2.0', '2.5', '>3']
    ylabels = [str(item) for item in y]
    plt.imshow(recall, cmap="RdPu", vmin=0.0, vmax=1.0)
    plt.xlabel('Delta F / F')
    plt.ylabel('Models')
    plt.xticks(ticks=x, labels=xlabels)
    plt.yticks(ticks=y, labels=ylabels)
    plt.colorbar(orientation="vertical") 
    # fig.savefig(f"./figures/TPFPFN_matrices/{args.model}_recall_matrix_binned.png", bbox_inches='tight')
    fig.savefig(f"./figures/TPFPFN_matrices/{args.model}_recall_matrix_binned.pdf", transparent=True, bbox_inches='tight')
    plt.close(fig)

def plot_precision_matrix(precision: np.ndarray) -> None:
    fig = plt.figure()
    x = np.arange(0, precision.shape[1], 1)
    y = np.arange(0, precision.shape[0], 1)
    xlabels = ['0.0', '0.5', '1.0', '1.5', '2.0', '2.5', '>3']
    ylabels = [str(item) for item in y]
    plt.imshow(precision, cmap="RdPu", vmin=0.0, vmax=1.0)
    plt.xlabel('Delta F / F')
    plt.xticks(ticks=x, labels=xlabels)
    plt.yticks(ticks=y, labels=ylabels)
    plt.ylabel('Models')
    plt.colorbar(orientation="vertical") 
    # fig.savefig(f"./figures/TPFPFN_matrices/{args.model}_precision_matrix.png", bbox_inches='tight')
    fig.savefig(f"./figures/TPFPFN_matrices/{args.model}_precision_matrix_binned.pdf", transparent=True, bbox_inches='tight')
    plt.close(fig)

def plot_accuracy_matrix(accuracy: np.ndarray) -> None:
    fig = plt.figure()
    x = np.arange(0, accuracy.shape[1], 1)
    y = np.arange(0, accuracy.shape[0], 1)
    xlabels = ['0.0', '0.5', '1.0', '1.5', '2.0', '2.5', '>3']
    ylabels = [str(item) for item in y]
    plt.imshow(accuracy, cmap="RdPu", vmin=0.0, vmax=1.0)
    plt.xlabel('Delta F / F')
    plt.xticks(ticks=x, labels=xlabels)
    plt.yticks(ticks=y, labels=ylabels)
    plt.ylabel('Models')
    plt.colorbar(orientation="vertical") 
    # fig.savefig(f"./figures/TPFPFN_matrices/{args.model}_accuracy_matrix.png", bbox_inches='tight')
    fig.savefig(f"./figures/TPFPFN_matrices/{args.model}_accuracy_matrix_binned.pdf", transparent=True, bbox_inches='tight')
    plt.close(fig)

def elementwise_recall_and_precision(tp_array: np.ndarray, fp_array: np.ndarray, fn_array: np.ndarray) -> np.ndarray:
    recall_array = tp_array/(tp_array + fn_array)
    precision_array = tp_array/(tp_array+fp_array)
    return recall_array, precision_array

def plot_total_numbers(summed_array: np.ndarray) -> None:
    fig = plt.figure()
    plt.imshow(summed_array, cmap="RdPu")
    plt.xlabel('Delta F / F')
    plt.ylabel('Models')
    plt.colorbar(orientation="vertical") 
    # fig.savefig(f"./figures/TPFPFN_matrices/{args.model}_total_events.png", bbox_inches='tight')
    fig.savefig(f"./figures/TPFPFN_matrices/{args.model}_total_events.pdf", transparent=True, bbox_inches='tight')
    plt.close(fig)

def get_data_path():
    if args.model == "UNet":
        return "./UNet__tp_fp_fn_data.npz"
    elif args.model == "StarDist":
        return "./SD__tp_fp_fn_data.npz"
    elif args.model == "IDT":
        return "./IDT_tp_fp_fn.npz"
    else:
        exit("No such results")

def load_data(model: str):
    path = get_data_path()
    data = np.load(path)
    tp_array, fp_array, fn_array = data["tp_array"], data["fp_array"], data["fn_array"]
    if args.model == "IDT":
        tp_array = np.reshape(tp_array, newshape=(1, 13))
        fp_array = np.reshape(fp_array, newshape=(1, 13))
        fn_array = np.reshape(fn_array, newshape=(1, 13))
    tp_array, fp_array, fn_array = bin_intense_events(
        tp_array=tp_array,
        fp_array=fp_array,
        fn_array=fn_array
    )
    stacked = np.stack([tp_array, fp_array, fn_array], axis=0)
    summed = np.sum(stacked, axis=0)
    # plot_total_numbers(summed)
    accuracy = tp_array / summed
    # fp_array = fp_array / summed
    # fn_array = fn_array / summed
    return tp_array, fp_array, fn_array, accuracy

def bin_intense_events(tp_array: np.ndarray, fp_array: np.ndarray, fn_array: np.ndarray, threshold_idx: int = 6) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    new_tp = np.zeros((tp_array.shape[0], threshold_idx+1))
    new_fp = np.zeros((fp_array.shape[0], threshold_idx+1))
    new_fn = np.zeros((fn_array.shape[0], threshold_idx+1))
    new_tp[:, :threshold_idx] = tp_array[:, :threshold_idx]
    new_fp[:, :threshold_idx] = fp_array[:, :threshold_idx]
    new_fn[:, :threshold_idx] = fn_array[:, :threshold_idx]
    binned_tp = np.sum(tp_array[:, threshold_idx:], axis=1)
    binned_fp = np.sum(fp_array[:, threshold_idx:], axis=1)
    binned_fn = np.sum(fn_array[:, threshold_idx:], axis=1)
    new_tp[:, -1] = binned_tp
    new_fp[:, -1] = binned_fp
    new_fn[:, -1] = binned_fn
    return new_tp, new_fp, new_fn


def main():
    tp_array, fp_array, fn_array, accuracy = load_data(model=args.model)
    # tp_array, fp_array, fn_array, accuracy = tp_array[:, :9], fp_array[:, :9], fn_array[:, :9], accuracy[:, :9] 
    recall_array, precision_array = elementwise_recall_and_precision(tp_array, fp_array, fn_array)
    plot_accuracy_matrix(accuracy)
    plot_recall_matrix(recall=recall_array)
    plot_precision_matrix(precision=precision_array)
    # plot_fancy_histogram(tp_array, "TP", rate=True)
    # plot_fancy_histogram(fp_array, "FP", rate=True)
    # plot_fancy_histogram(fn_array, "FN", rate=True)

if __name__=="__main__":
    main()
