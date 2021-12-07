import os
import matplotlib.pyplot as plt

from visualization.slice_selection import select_slice
from visualization.heatmap import heatmap_plot


def save_sample(epoch, opt, real_dm, real_gas, pred_gas):
    slices = select_slice(real_dm, real_gas, pred_gas, random_dims=(0, 1), orthogonal_dim=2, weight=0.1)
    fig, ax = heatmap_plot(*[s.squeeze() for s in slices], subplot_titles=("dark matter", "real gas", "predicted gas"))

    filename = os.path.join(opt.run_path, f"samples_{opt.run_index}", f"E{epoch:03d}.png")
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename)
    plt.close()