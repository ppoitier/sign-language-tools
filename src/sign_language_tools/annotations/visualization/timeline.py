from typing import Any

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes


__all__ = [
    "plot_segments_on_timeline",
]


def plot_segments_on_timeline(
    segments: np.ndarray,
    labels: list[str] = None,
    y_lim: tuple[float, float] = (0, 0.5),
    ax: Axes | None = None,
    alpha: float = 0.5,
    colors: Any = None,
    cmap: str = "tab20",
    **kwargs: Any,
):
    """Plot segments as colored spans on a timeline.

    Each segment is drawn as a vertical span between its start and end
    frame. If `labels` is given, the label of each segment is drawn
    centered in its span. Colors can be assigned automatically (one
    color per unique label), given as a single color applied to every
    segment, or given explicitly per segment.

    Args:
        segments: Array of shape `(M, 2)` or `(M, 3)`, where `M` is the
            number of segments and columns are `[start, end]` or
            `[start, end, label]`, with an inclusive end.
        labels: Optional list of `M` labels, one per segment. Segments
            with a `None` label are drawn without text.
        y_lim: Tuple `(y_min, y_max)` giving the vertical extent of each
            span, as axis-relative coordinates in `[0, 1]`.
        ax: Matplotlib axes to draw on. Defaults to the current axes.
        alpha: Opacity of the segment spans.
        colors: Colors to use for the segments. Can be:

            - `None` (default): a color per unique label is picked
              automatically from `cmap` if `labels` is given, otherwise
              matplotlib's default color is used for every segment.
            - A single color, applied to every segment.
            - A list of `M` colors, one per segment.
        cmap: Name of the matplotlib colormap used for automatic
            per-label coloring. Ignored if `colors` is not `None`.
        **kwargs: Additional keyword arguments forwarded to
            `Axes.axvspan`.

    Example:
        >>> import numpy as np
        >>> segments = np.array([[0, 10], [15, 25]])
        >>> plot_segments_on_timeline(segments, labels=["a", "b"])
    """
    if ax is None:
        ax = plt.gca()

    if colors is None and labels is not None:
        # Auto-assign a color per unique label.
        unique = sorted(set(lab for lab in labels if lab is not None))
        cmap_obj = plt.get_cmap(cmap)
        label_to_color = {lab: cmap_obj(i % cmap_obj.N) for i, lab in enumerate(unique)}
        colors = [label_to_color[lab] if lab is not None else None for lab in labels]
    elif colors is None or isinstance(colors, str):
        colors = [colors] * len(segments)
    elif len(colors) != len(segments):
        raise ValueError(
            f"`colors` must have one entry per segment: got {len(colors)} colors "
            f"for {len(segments)} segments."
        )

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
