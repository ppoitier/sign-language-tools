import numpy as np
import matplotlib.pyplot as plt


def plot_segments(segments: np.ndarray, x_lim=None):
    fig, ax = plt.subplots(1, 1, figsize=(30, 2))

    if x_lim is not None:
        ax.set_xlim(*x_lim)

    for seg in segments[:, :2].astype('float32'):
        ax.axvspan(seg[0], seg[1], alpha=.5)

    fig.show()
