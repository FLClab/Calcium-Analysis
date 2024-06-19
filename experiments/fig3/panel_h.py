import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib 
from typing import List 
cmap_og = matplotlib.colors.LinearSegmentedColormap.from_list(
    name="nice-prism",
    colors=["#5F4690","#1D6996","#38A6A5","#0F8554","#73AF48","#EDAD08","#E17C05","#CC503E","#94346E"]
)
matplotlib.colormaps.register(cmap=cmap_og, force=True)
matplotlib.colormaps.register(cmap=cmap_og.reversed(), force=True)

cmap = plt.get_cmap('nice-prism', 11)


def compute_f1(data, model_id: int, stop: int = 3):
    tp = np.sum(data["tp_array"][model_id, :])
    fp = np.sum(data["fp_array"][model_id, :])
    fn = np.sum(data["fn_array"][model_id, :])

    print(f"Full events\n\tTP: {tp}, FP: {fp}, FN: {fn}")

    f1 = (2 * tp) / ((2 * tp) + fp + fn)

    tp_dim = np.sum(data["tp_array"][model_id, :2])
    fp_dim = np.sum(data["fp_array"][model_id, :2])
    fn_dim = np.sum(data["fn_array"][model_id, :2])
    print(f"Dim events\n\tTP: {tp_dim}, FP: {fp_dim}, FN: {fn_dim}\n")
    f1_dim = (2 * tp_dim) / ((2 * tp_dim) + fp_dim + fn_dim)
    return f1, f1_dim

def load_all_data():
    idt = np.load("./IDT_tp_fp_fn.npz")
    idt_tp = np.sum(idt["tp_array"])
    idt_fp = np.sum(idt["fp_array"])
    idt_fn = np.sum(idt["fn_array"])
    unet_data = np.load("./UNet__tp_fp_fn_data.npz")
    sd_data = np.load("./SD__tp_fp_fn_data.npz")
    sd_f1, sd_dim_f1 = compute_f1(data=sd_data, model_id=0)

    unet_f1, unet_dim_f1 = compute_f1(data=unet_data, model_id=1)
    unet64_f1, unet64_dim_f1 = compute_f1(data=unet_data, model_id=8)
    return (sd_f1, sd_dim_f1), (unet_f1, unet_dim_f1), (unet64_f1, unet64_dim_f1)

def make_plot(stardist, unet, unet64):
    pass


def main():
    stardist_f1, unet_f1, unet64_f1 = load_all_data()
    print(stardist_f1)
    print(unet_f1)
    print(unet64_f1)
    exit()

if __name__=="__main__":
    main()
