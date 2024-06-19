import numpy as np
import matplotlib.pyplot as plt
from typing import List
import matplotlib
from typing import Tuple
cmap_og = matplotlib.colors.LinearSegmentedColormap.from_list(
    name="nice-prism",
    colors=["#5F4690","#1D6996","#38A6A5","#0F8554","#73AF48","#EDAD08","#E17C05","#CC503E","#94346E"]
)
matplotlib.colormaps.register(cmap=cmap_og, force=True)
matplotlib.colormaps.register(cmap=cmap_og.reversed(), force=True)

cmap = plt.get_cmap('nice-prism', 11)

def load_model_FNs() -> List[int]:
    idt = np.load("./IDT_tp_fp_fn.npz")['fn_array']
    stardist = np.load("./SD__tp_fp_fn_data.npz")['fn_array']
    unet = np.load("./UNet__tp_fp_fn_data.npz")['fn_array']
    missed_idt = np.sum(idt[:3])
    missed_stardist = np.sum(stardist[0, :3])
    missed_unet = np.sum(unet[8, :3])
    return [missed_idt, missed_stardist, missed_unet]


def load_model_FPs() -> List[int]:
    idt = np.load("./IDT_tp_fp_fn.npz")['fp_array']
    stardist = np.load("./SD__tp_fp_fn_data.npz")['fp_array']
    unet = np.load("./UNet__tp_fp_fn_data.npz")['fp_array']
    missed_idt = np.sum(idt[:3])
    missed_stardist = np.sum(stardist[0, :3])
    missed_unet = np.sum(unet[8, :3])
    return [missed_idt, missed_stardist, missed_unet]

def load_model_TPs() -> List[int]:
    idt = np.load("./IDT_tp_fp_fn.npz")['tp_array']
    stardist = np.load("./SD__tp_fp_fn_data.npz")['tp_array']
    unet = np.load("./UNet__tp_fp_fn_data.npz")['tp_array']
    missed_idt = np.sum(idt[:3])
    missed_stardist = np.sum(stardist[0, :3])
    missed_unet = np.sum(unet[8, :3])
    return [missed_idt, missed_stardist, missed_unet]

def plot_FNs(data: List[int]) ->  None:
    bars = ['IDT', "StarDist-3D", "3D U-Net"]
    x = np.arange(len(bars))
    fig = plt.figure()
    plt.bar(x, data, color='gray', edgecolor='black')
    plt.xticks(ticks=x, labels=bars)
    plt.xlabel("Models")
    plt.ylabel("# Events missed")
    fig.savefig("./figures/missed/missed_events.pdf", transparent=True, bbox_inches='tight')

def plot_FPs(data: List[int]) ->  None:
    bars = ['IDT', "StarDist-3D", "3D U-Net"]
    x = np.arange(len(bars))
    fig = plt.figure()
    plt.bar(x, data, color='gray', edgecolor='black')
    plt.xticks(ticks=x, labels=bars)
    plt.xlabel("Models")
    plt.ylabel("# Events hallucinated")
    fig.savefig("./figures/missed/hallucinated_events.pdf", transparent=True, bbox_inches='tight')

def plot_TPs(data: List[int]) ->  None:
    bars = ['IDT', "StarDist-3D", "3D U-Net"]
    x = np.arange(len(bars))
    fig = plt.figure()
    plt.bar(x, data, color='gray', edgecolor='black')
    plt.xticks(ticks=x, labels=bars)
    plt.xlabel("Models")
    plt.ylabel("# Events found")
    fig.savefig("./figures/missed/found_events.pdf", transparent=True, bbox_inches='tight')

def load_deep_models_data(model_id: int):
    stardist = np.load("./SD__tp_fp_fn_data.npz")['fn_array']
    unet = np.load("./UNet__tp_fp_fn_data.npz")['fn_array']
    missed_stardist = np.sum(stardist[model_id, :3])
    missed_unet = np.sum(unet[model_id, :3])
    return (missed_stardist, missed_unet)

def beeswarm_missed_events(idt: int, stardist: list, unet: list) -> None:
    xs = [0, 1]
    variance = 0.08
    boxplot_data = np.zeros((len(stardist), 2))
    boxplot_data[:, 0] = stardist
    boxplot_data[:, 1] = unet
    fig = plt.figure()
    plt.axhline(y=idt, xmin=0, xmax=1, color='black', ls='--', label="IDT")
    # plt.bar(x=0, height=idt, width=0.25, color='gray', edgecolor='black')
    plt.boxplot(x=boxplot_data, positions=xs, showfliers=False, medianprops={"color": "gray"})
    for i in range(11):
        x_sd = np.random.normal(xs[0], scale=variance)
        x_unet = np.random.normal(xs[1], scale=variance)
        c = "black" if i == 0 else cmap(i-1)
        plt.scatter(x_sd, stardist[i], color=c, s=70)
        plt.scatter(x_unet, unet[i], color=c, s=70)
    plt.ylabel("# Missed events")
    plt.xticks(ticks=[0, 1], labels=["StarDist-3D", "3D U-Net"])
    plt.legend()
    fig.savefig("./figures/missed/beeswarm_missed_events.pdf", transparent=True, bbox_inches='tight')

def beeswarm_f1(idt: int, stardist: list, unet: list) -> None:
    xs = [0, 1]
    variance = 0.08
    boxplot_data = np.zeros((len(stardist), 2))
    boxplot_data[:, 0] = stardist
    boxplot_data[:, 1] = unet
    fig = plt.figure()
    plt.axhline(y=idt, xmin=0, xmax=1, color='black', ls='--', label="IDT")
    # plt.bar(x=0, height=idt, width=0.25, color='gray', edgecolor='black')
    plt.boxplot(x=boxplot_data, positions=xs, showfliers=False, medianprops={"color": "gray"})
    for i in range(11):
        x_sd = np.random.normal(xs[0], scale=variance)
        x_unet = np.random.normal(xs[1], scale=variance)
        c = "black" if i == 0 else cmap(i-1)
        plt.scatter(x_sd, stardist[i], color=c, s=70)
        plt.scatter(x_unet, unet[i], color=c, s=70)
    plt.ylabel("F1-score")
    plt.xticks(ticks=[0, 1.0], labels=["StarDist-3D", "3D U-Net"])
    plt.legend()
    fig.savefig("./figures/missed/beeswarm_f1-score.pdf", transparent=True, bbox_inches='tight')

def compute_f1(data, model_id: int) -> np.ndarray:
    tp = np.sum(data["tp_array"][model_id, :])
    fp = np.sum(data["fp_array"][model_id, :])
    fn = np.sum(data["fn_array"][model_id, :])
    return (2 * tp) / ((2 * tp) + fp + fn)

def data_to_f1() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    idt = np.load("./IDT_tp_fp_fn.npz")
    idt_tp = np.sum(idt["tp_array"])
    idt_fp = np.sum(idt["fp_array"])
    idt_fn = np.sum(idt["fn_array"])
    idt_f1 = (2 *idt_tp) / ((2*idt_tp) + idt_fp + idt_fn)

    stardist_f1, unet_f1 = [], []
    stardist_data = np.load("./SD__tp_fp_fn_data.npz")
    unet_data = np.load("./UNet__tp_fp_fn_data.npz")
    for i in range(11):
        sd_f1 = compute_f1(data=stardist_data, model_id=i)
        u_f1 = compute_f1(data=unet_data, model_id=i)
        stardist_f1.append(sd_f1)
        unet_f1.append(u_f1)
    return (idt_f1, stardist_f1, unet_f1)

def load_best_data():
    idt = np.load("./IDT_tp_fp_fn.npz")
    idt_tp = np.sum(idt["tp_array"])
    idt_fp = np.sum(idt["fp_array"])
    idt_fn = np.sum(idt["fn_array"])
    idt_f1 = (2 *idt_tp) / ((2*idt_tp) + idt_fp + idt_fn)
    stardist_f1, unet_f1 = [], []
    stardist_data = np.load("./TPFPFN/StarDist-4-0__tp_fp_fn_data.npz")
    unet_data = np.load("./TPFPFN/UNet-64-0__tp_fp_fn_data.npz")
    for i in range(25):
        # sd_f1 = compute_f1(data=stardist_data, model_id=i)
        u_f1 = compute_f1(data=unet_data, model_id=i)
        # stardist_f1.append(sd_f1)
        unet_f1.append(u_f1)
        print(u_f1)
        # print(sd_f1, u_f1)
    return (idt_f1, stardist_f1, unet_f1)

    

def main():
    ## To compute the number of missed events per best model as a bar plot
    # fps = load_model_FPs()
    # fns = load_model_FNs()
    # tps = load_model_TPs()
    # plot_FNs(data=fns)
    # plot_FPs(data=fps)
    # plot_TPs(data=tps)
    load_best_data()
    exit()

    # To compute the number of missed events for all PU configs of the models as a beeswarm
    idt = np.load("./IDT_tp_fp_fn.npz")['fn_array']
    idt_data = np.sum(idt[:3])
    stardist_data = []
    unet_data = []
    for i in range(11): # 11 different PU configurations
        sd, unet = load_deep_models_data(model_id=i)
        stardist_data.append(sd)
        unet_data.append(unet)
    beeswarm_missed_events(
        idt=idt_data,
        stardist=stardist_data,
        unet=unet_data
    )

    ## To compute the F1-score for all PU configs of the models as a beeswarm
    idt_f1, stardist_f1, unet_f1 = data_to_f1()
    beeswarm_f1(
        idt=idt_f1,
        stardist=stardist_f1,
        unet=unet_f1
    )

    ## To compute the F1-score for all seeds of the models' best PU config, as a beeswarm
    # load_best_data()



if __name__=="__main__":
    main()


