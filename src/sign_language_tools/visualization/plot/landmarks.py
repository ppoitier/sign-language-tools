from typing import Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.lines import Line2D


__all__ = [
    'plot_landmarks',
]

Edge = Tuple[int, int]


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
        aspect_ratio: float = 1,
        show_axis: bool = False,
        show_indices: bool = False,
        x_lim: Optional[tuple[float, float]] = None,
        y_lim: Optional[tuple[float, float]] = None,
        refocus: bool = False,
        focus_pad: float = 0.02,
):
    """
    Plot landmarks on a figure using matplotlib.
    Landmarks have to be a Numpy array of shape (L, 2) or (L, 3) where L is the number of different landmarks.

    Args:
        landmarks: The set of landmarks (L, 2) or (L, 3) to plot. Each landmark is represented as a vertex.
        connections: An optional tuple of edges to draw between vertices.
        ax: If specified, this matplotlib axis will be used. Otherwise, global axis are used.
        vertex_size: Size of all vertices. Default=0.01
        vertex_color: Color of all vertices. Default='lime'
        edge_color: Color of all edges. Default='white'
        background_color: Color of the background. Default='black'
        text_color: Color of the text, if present. Default='red'
        aspect_ratio: Aspect ratio used to display the landmarks. Default=1.0
        show_axis: If true, draw axis. Default=False
        show_indices: If true, draw indices of landmarks. Default=False
        x_lim: If specified, override computed x_lim. Default=None
        y_lim: If specified, override computed x_lim. Default=None
        refocus: If true, compute x- and y-lim to refocus on landmarks. Default=False
        focus_pad: Specify padding used if refocus is true. Otherwise, it is ignored.

    Author:
        v0.0.1 - ppoitier
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
    ax.set_box_aspect(1/aspect_ratio)
    ax.set_xlim(*x_lim)
    ax.set_ylim(*y_lim)

    _draw_vertices(landmarks, ax=ax, vertex_size=vertex_size, vertex_color=vertex_color)
    if show_indices:
        _draw_indices(landmarks, ax=ax, color=text_color)
    if connections is not None:
        _draw_edges(x, y, ax=ax, connections=connections, edge_color=edge_color)

    return ax


# ---- Helpers


def _compute_refocus(
        landmarks: np.ndarray,
        focus_pad: float,
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """
    Calculate axis limits to refocus the figure on landmarks, with configurable padding.

    Args:
        landmarks: Set of landmarks (L, 2) or (L, 3) on which to refocus the figure.
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


def _compute_xy_lim(
        landmarks: np.ndarray,
        aspect_ratio: float,
        *,
        x_lim: Optional[float] = None,
        y_lim: Optional[float] = None,
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

    x_size = x_lim[1] - x_lim[0]
    y_size = y_lim[0] - y_lim[1]
    aspect_ratio = (x_size * aspect_ratio / y_size)

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
        ax.add_patch(Circle(
            (coords[0], coords[1]),
            radius=vertex_size/2,
            facecolor=vertex_color,
            zorder=2,
        ))


def _draw_indices(landmarks: np.ndarray, ax, color):
    for index, coords in enumerate(landmarks):
        ax.text(coords[0], coords[1], str(index), color=color, clip_on=True)
