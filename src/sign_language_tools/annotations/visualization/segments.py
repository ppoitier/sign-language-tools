import numpy as np
import matplotlib.pyplot as plt

# def plot_segments(segments: np.ndarray, labels: list[str] = None, y_lim=(0, 0.5), ax=None, alpha=0.5, **kwargs):
#     if ax is None:
#         ax = plt.gca()
#
#     y_min, y_max = y_lim
#     for index, segment in enumerate(segments):
#         start, end = segment[:2]
#         ax.axvspan(start, end+1, ymin=y_min, ymax=y_max, alpha=alpha, edgecolor='black', linewidth=0.5, **kwargs)
#         if labels is not None:
#             label = labels[index]
#             ax.text(start + (end - start) / 2, 0.5, str(label), ha='center', rotation='vertical', va='center')


def plot_segments(
    segments: np.ndarray,
    labels: list[str] = None,
    y_lim=(0, 0.5),
    ax=None,
    alpha=0.5,
    colors=None,
    cmap="tab20",
    **kwargs,
):
    if ax is None:
        ax = plt.gca()

    # Auto-assign a color per unique label if no explicit colors given
    if colors is None and labels is not None:
        unique = sorted(set(lab for lab in labels if lab is not None))
        cmap_obj = plt.get_cmap(cmap)
        label_to_color = {lab: cmap_obj(i % cmap_obj.N) for i, lab in enumerate(unique)}
        colors = [label_to_color[lab] if lab is not None else None for lab in labels]
    elif colors is None or isinstance(colors, str):
        colors = [colors] * len(segments)

    y_min, y_max = y_lim
    for index, segment in enumerate(segments):
        start, end = segment[:2]
        ax.axvspan(
            start,
            end + 1,
            ymin=y_min,
            ymax=y_max,
            alpha=alpha,
            edgecolor="black",
            linewidth=0.5,
            facecolor=colors[index],
            **kwargs,
        )
        if labels is not None:
            label = labels[index]
            if label is not None:
                ax.text(
                    start + (end - start) / 2,
                    0.5,
                    str(label),
                    ha="center",
                    rotation="horizontal",
                    va="center",
                    clip_on=True,
                )
