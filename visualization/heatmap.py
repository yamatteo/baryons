from typing import Sequence

import matplotlib.pyplot as plt
import seaborn as sb


def heatmap_plot(*tensors, title: str = None, subplot_titles: Sequence[str] = None):
    assert all(t.dim() == 2 for t in tensors), "All tensors must be 2D"
    assert all(t.shape == tensors[0].shape for t in tensors), "All tensors must have the same shape"

    if subplot_titles is None:
        subplot_titles = [str(i) for i in range(len(tensors))]

    fig, axs = plt.subplots(1, len(tensors))
    fig.suptitle = title

    for i, t in enumerate(tensors):
        sb.heatmap(t, ax=axs[i], cbar=False, square=True, xticklabels=False, yticklabels=False)
        try:
            axs[i].set_title(subplot_titles[i])
        except (TypeError, IndexError):
            pass

    return fig, axs
