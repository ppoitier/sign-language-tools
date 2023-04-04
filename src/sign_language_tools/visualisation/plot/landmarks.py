from typing import Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.lines import Line2D


Edge = Tuple[int, int]


def compute_refocus(
        landmarks: np.ndarray,
        focus_pad: float,
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """
    Compute the bounds of the frame that contains all the landmarks with a padding.

    Args:
        landmarks: Features at an instant t that contains a flat representation of the landmarks that are contained in
            the refocused frame.
        focus_pad: The padding used for the refocus.

    Returns:
        x_lim, y_lim : Boundaries of the refocused frame.
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


def compute_xy_lim(
        landmarks: np.ndarray,
        aspect_ratio: float,
        *,
        x_lim: Optional[float] = None,
        y_lim: Optional[float] = None,
        refocus: bool = False,
        focus_pad: float = 0.02,
):
    if refocus:
        refocus_x, refocus_y = compute_refocus(landmarks, focus_pad)
    else:
        refocus_x, refocus_y = (0, 1), (1, 0)

    if x_lim is None:
        x_lim = refocus_x
    if y_lim is None:
        y_lim = refocus_y

    x_size = x_lim[1] - x_lim[0]
    y_size = y_lim[0] - y_lim[1]
    aspect_ratio = (x_size * aspect_ratio / y_size)

    return x_lim, y_lim, aspect_ratio


def draw_edges(x, y, *, ax, connections, edge_color):
    for v0, v1 in connections:
        line = Line2D(
            [x[v0], x[v1]],
            [y[v0], y[v1]],
            color=edge_color,
            zorder=1,
        )
        ax.add_line(line)


def draw_vertices(landmarks: np.ndarray, ax, vertex_size, vertex_color):
    for coords in landmarks:
        ax.add_patch(Circle(
            (coords[0], coords[1]),
            radius=vertex_size/2,
            facecolor=vertex_color,
            zorder=2,
        ))


def draw_indices(landmarks: np.ndarray, ax, color):
    for index, coords in enumerate(landmarks):
        ax.text(coords[0], coords[1], str(index), color=color, clip_on=True)


def plot_landmarks(
        landmarks: np.ndarray,
        connections: Optional[Tuple[Edge, ...]] = None,
        *,
        ax=None,
        vertex_size: float = 0.01,
        vertex_color='lime',
        edge_color='white',
        background_color='black',
        text_color='red',
        aspect_ratio=1,
        show_axis: bool = False,
        show_indices: bool = False,
        x_lim: Optional[tuple[float, float]] = None,
        y_lim: Optional[tuple[float, float]] = None,
        refocus: bool = False,
        focus_pad: float = 0.02,
):
    x = landmarks[:, 0]
    y = landmarks[:, 1]

    x_lim, y_lim, aspect_ratio = compute_xy_lim(
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
    ax.set_box_aspect(1/aspect_ratio)
    ax.set_xlim(*x_lim)
    ax.set_ylim(*y_lim)

    draw_vertices(landmarks, ax=ax, vertex_size=vertex_size, vertex_color=vertex_color)
    if show_indices:
        draw_indices(landmarks, ax=ax, color=text_color)
    if connections is not None:
        draw_edges(x, y, ax=ax, connections=connections, edge_color=edge_color)

    return ax
