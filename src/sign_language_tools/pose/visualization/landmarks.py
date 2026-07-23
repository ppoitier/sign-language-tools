from typing import Any, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.lines import Line2D


__all__ = [
    "plot_landmarks",
    "plot_landmarks_sequence",
]

Edge = Tuple[int, int]


def plot_landmarks(
    landmarks: np.ndarray,
    connections: Optional[Tuple[Edge, ...]] = None,
    *,
    ax=None,
    vertex_size: float = 0.01,
    vertex_color: str = "lime",
    edge_color: str = "white",
    background_color: str = "black",
    text_color: str = "red",
    aspect_ratio: float = 1,
    show_axis: bool = False,
    show_indices: bool = False,
    x_lim: Optional[Tuple[float, float]] = None,
    y_lim: Optional[Tuple[float, float]] = None,
    refocus: bool = False,
    focus_pad: float = 0.02,
):
    """Plots landmarks on a matplotlib axis.

    Args:
        landmarks: Landmarks to plot, of shape `(L, 2)` or `(L, 3)`, where
            `L` is the number of landmarks. Each landmark is drawn as a
            vertex.
        connections: Edges to draw between vertices, as `(start, end)`
            index pairs. If `None`, no edges are drawn.
        ax: Matplotlib axes to draw on. Defaults to the current axes.
        vertex_size: Diameter of each vertex marker.
        vertex_color: Color of the vertex markers.
        edge_color: Color of the edges.
        background_color: Color of the axes background.
        text_color: Color of the landmark indices, if shown.
        aspect_ratio: Aspect ratio (width / height) used to display the
            landmarks.
        show_axis: Whether to draw the x- and y-axis.
        show_indices: Whether to draw the index of each landmark next to
            its vertex.
        x_lim: Override for the computed x-axis limits.
        y_lim: Override for the computed y-axis limits.
        refocus: If `True`, compute `x_lim`/`y_lim` to fit tightly around
            the landmarks instead of using the default `[0, 1]` unit
            square.
        focus_pad: Padding added around the landmarks when `refocus` is
            `True`. Ignored otherwise.

    Returns:
        The matplotlib axes the landmarks were drawn on.

    Example:
        >>> import numpy as np
        >>> from sign_language_tools.pose.visualization import plot_landmarks
        >>> landmarks = np.random.rand(21, 2)
        >>> ax = plot_landmarks(landmarks)
    """
    x = landmarks[:, 0]
    y = landmarks[:, 1]

    x_lim, y_lim, aspect_ratio = _compute_xy_lim(
        landmarks,
        aspect_ratio=aspect_ratio,
        x_lim=x_lim,
        y_lim=y_lim,
        refocus=refocus,
        focus_pad=focus_pad,
    )

    if ax is None:
        ax = plt.gca()
    ax.set_facecolor(background_color)
    ax.axes.xaxis.set_visible(show_axis)
    ax.axes.yaxis.set_visible(show_axis)
    ax.set_box_aspect(1 / aspect_ratio)
    ax.set_xlim(*x_lim)
    ax.set_ylim(*y_lim)

    _draw_vertices(landmarks, ax=ax, vertex_size=vertex_size, vertex_color=vertex_color)
    if show_indices:
        _draw_indices(landmarks, ax=ax, color=text_color)
    if connections is not None:
        _draw_edges(x, y, ax=ax, connections=connections, edge_color=edge_color)

    return ax


def plot_landmarks_sequence(
    landmarks: dict[Any, np.ndarray],
    connections: Optional[dict[Any, Tuple[Edge, ...]]] = None,
    **kwargs,
):
    """Plots a temporal sequence of landmark groups, one subplot per frame.

    Args:
        landmarks: Mapping from a group name (e.g. `"left_hand"`) to its
            landmark sequence, of shape `(T, L, 2)` or `(T, L, 3)`. All
            groups must share the same sequence length `T`.
        connections: Mapping from a group name to the edges drawn for
            that group, as in [`plot_landmarks`][sign_language_tools.pose.visualization.plot_landmarks].
            Groups without an entry are drawn without edges.
        **kwargs: Additional keyword arguments forwarded to
            [`plot_landmarks`][sign_language_tools.pose.visualization.plot_landmarks]
            for every vertex group and every frame.

    Returns:
        The matplotlib figure containing one subplot per frame.

    Example:
        >>> import numpy as np
        >>> from sign_language_tools.pose.visualization import plot_landmarks_sequence
        >>> landmarks = {"right_hand": np.random.rand(10, 21, 2)}
        >>> fig = plot_landmarks_sequence(landmarks)
    """
    connections = connections or {}
    sequence_len = len(next(iter(landmarks.values())))

    fig, subplots = plt.subplots(1, sequence_len)

    for frame_index in range(sequence_len):
        ax = subplots[frame_index]
        for group_name, group_landmarks in landmarks.items():
            ax = plot_landmarks(
                group_landmarks[frame_index],
                connections.get(group_name),
                ax=ax,
                **kwargs,
            )

    return fig


# ---- Helpers


def _compute_refocus(
    landmarks: np.ndarray,
    focus_pad: float,
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """Computes axis limits that tightly frame `landmarks`, with padding.

    Args:
        landmarks: Landmarks, of shape `(L, 2)` or `(L, 3)`, to refocus on.
        focus_pad: Padding added around the landmarks.

    Returns:
        The `x_lim` and `y_lim` boundaries of the refocused frame.
    """
    x = landmarks[:, 0]
    y = landmarks[:, 1]

    x_min = np.min(x)
    x_max = np.max(x)

    y_min = np.min(y)
    y_max = np.max(y)

    x_lim = (x_min - focus_pad, x_max + focus_pad)
    y_lim = (y_max + focus_pad, y_min - focus_pad)

    return x_lim, y_lim


def _compute_xy_lim(
    landmarks: np.ndarray,
    aspect_ratio: float,
    *,
    x_lim: Optional[Tuple[float, float]] = None,
    y_lim: Optional[Tuple[float, float]] = None,
    refocus: bool = False,
    focus_pad: float = 0.02,
):
    if refocus:
        refocus_x, refocus_y = _compute_refocus(landmarks, focus_pad)
    else:
        refocus_x, refocus_y = (0, 1), (1, 0)

    if x_lim is None:
        x_lim = refocus_x
    if y_lim is None:
        y_lim = refocus_y

    x_size = abs(x_lim[1] - x_lim[0])
    y_size = abs(y_lim[0] - y_lim[1])
    aspect_ratio = x_size * aspect_ratio / y_size

    return x_lim, y_lim, aspect_ratio


def _draw_edges(x, y, *, ax, connections, edge_color):
    for v0, v1 in connections:
        line = Line2D(
            [x[v0], x[v1]],
            [y[v0], y[v1]],
            color=edge_color,
            zorder=1,
        )
        ax.add_line(line)


def _draw_vertices(landmarks: np.ndarray, ax, vertex_size, vertex_color):
    for coords in landmarks:
        ax.add_patch(
            Circle(
                (coords[0], coords[1]),
                radius=vertex_size / 2,
                facecolor=vertex_color,
                zorder=2,
            )
        )


def _draw_indices(landmarks: np.ndarray, ax, color):
    for index, coords in enumerate(landmarks):
        ax.text(coords[0], coords[1], str(index), color=color, clip_on=True)
