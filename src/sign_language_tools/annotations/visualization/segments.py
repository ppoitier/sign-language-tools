import numpy as np
import matplotlib.pyplot as plt


def plot_segments(segments: np.ndarray, labels: list[str] = None, ax=None, alpha=0.5):
    if ax is None:
        ax = plt.gca()

    for index, segment in enumerate(segments):
        start, end = segment[:2]
        ax.axvspan(start, end, alpha=alpha)
        if labels is not None:
            label = labels[index]
            plt.text(start + (end - start) / 2, 0.5, str(label), ha='center')
