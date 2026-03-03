import numpy as np
import matplotlib.pyplot as plt


def plot_segments(segments: np.ndarray, labels: list[str] = None, y_lim=(0, 0.5), ax=None, alpha=0.5, **kwargs):
    if ax is None:
        ax = plt.gca()

    y_min, y_max = y_lim
    for index, segment in enumerate(segments):
        start, end = segment[:2]
        ax.axvspan(start, end+1, ymin=y_min, ymax=y_max, alpha=alpha, edgecolor='black', linewidth=0.5, **kwargs)
        if labels is not None:
            label = labels[index]
            ax.text(start + (end - start) / 2, 0.5, str(label), ha='center', rotation='vertical', va='center')
