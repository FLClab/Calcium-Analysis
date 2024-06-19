import numpy as np
import matplotlib.pyplot as plt
import seaborn
import pandas
from features_2d import load_data

cluster_dict = {
    1: ' Cluster 1',
    2: 'Cluster 2',
    3: 'Cluster 3',
    4: 'Cluster 4'
}

def build_ridgeline(data, feature):
    pal = seaborn.color_palette(palette='coolwarm', n_colors=4)
    cluster_mean_serie = data.groupby("Cluster")[feature].mean()
    data['mean_cluster'] = data['Cluster'].map(cluster_mean_serie)
    g = seaborn.FacetGrid(data, row='Cluster', hue='mean_cluster', aspect=15, height=0.75, palette=pal)
    g.map(seaborn.kdeplot, feature, bw_adjust=1, clip_on=False,fill=True, alpha=1, linewidth=1.5)
    g.map(seaborn.kdeplot, feature, bw_adjust=1, clip_on=False, color='white', lw=2)
    g.map(plt.axhline, y=0, lw=2, clip_on=False)

    for i, ax in enumerate(g.axes.flat):
        print(i)
        ax.text(-15, 0.02, cluster_dict[i + 1], fontweight='bold', fontsize=15, color=ax.lines[-1].get_color())
    
    g.fig.subplots_adjust(hspace=-0.3)
    g.set_titles("")
    g.set(yticks=[])
    g.despine(bottom=True, left=True)
    plt.setp(ax.get_xticklabels(), fontsize=15, fontweight='bold')
    plt.xlabel(feature, ha='right', fontsize=20, fontweight=20)
    plt.savefig('./temp.png')


def main():
    data = load_data()
    print(data)
    build_ridgeline(data, "Aspect ratio")

if __name__=="__main__":
    main()