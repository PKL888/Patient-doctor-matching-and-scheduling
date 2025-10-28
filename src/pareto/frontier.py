import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib import colors
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

plt.rcParams.update({
    "mathtext.fontset": "cm",   # Computer Modern for math
    "font.family": "serif",     # Serif font for text
    "font.size": 12             # Set default font size to 12
})


def plot_pareto_2d(pareto, dom=None, labels=("Pareto", "Dominated"), save_path="outputs/graphs"):
    """Plot 2D projections with optional dominated points."""
    obj0, obj1, obj2 = zip(*pareto) if pareto else ([], [], [])
    d_obj0, d_obj1, d_obj2 = zip(*dom) if dom else ([], [], [])

    cmap = plt.cm.get_cmap("RdYlGn")
    fig, axes = plt.subplots(1, 3, figsize=(15,5))

    axes[0].scatter(obj0, obj1, c=obj2, cmap=cmap, marker="o", label=labels[0])
    if dom: axes[0].scatter(d_obj0, d_obj1, c='grey', marker='x', label=labels[1])
    axes[0].set_xlabel("Objective 0: Patient satisfaction")
    axes[0].set_ylabel("Objective 1: Total appointments")
    axes[0].yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    axes[0].legend()

    axes[1].scatter(obj1, obj2, c=obj0, cmap=cmap, marker="o", label=labels[0])
    if dom: axes[1].scatter(d_obj1, d_obj2, c='grey', marker='x', label=labels[1])
    axes[1].set_xlabel("Objective 1: Total appointments")
    axes[1].set_ylabel("Objective 2: Doctor satisfaction")
    axes[1].xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    # axes[1].legend()

    axes[2].scatter(obj2, obj0, c=obj1, cmap=cmap, marker="o", label=labels[0])
    if dom: axes[2].scatter(d_obj2, d_obj0, c='grey', marker='x', label=labels[1])
    axes[2].set_xlabel("Objective 2: Doctor satisfaction")
    axes[2].set_ylabel("Objective 0: Patient satisfaction")
    # axes[2].legend()

    plt.suptitle("Pareto frontier with dominated points" if dom else "Pareto frontier")
    if save_path: plt.savefig(f"{save_path}/2d_plot.png", bbox_inches='tight', dpi=300)
    # plt.show()

def plot_pareto_3d(pareto, save_path="outputs/graphs"):
    """Plot 3D Pareto frontier."""
    fig = plt.figure(figsize=(8,7))
    ax = fig.add_subplot(111, projection='3d')
    cmap = plt.cm.get_cmap("RdYlGn")

    obj0, obj1, obj2 = zip(*pareto) if pareto else ([], [], [])
    ax.scatter(obj0, obj1, obj2, c=obj2, cmap=cmap, marker="o", alpha=1, s=80)

    ax.set_xlabel("Objective 0: Patient satisfaction")
    ax.set_ylabel("Objective 1: Total appointments")
    ax.set_zlabel("Objective 2: Doctor satisfaction")
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    plt.suptitle("Pareto frontier")
    if save_path: plt.savefig(f"{save_path}/3d_plot.png", bbox_inches='tight', dpi=300)
    # plt.show()