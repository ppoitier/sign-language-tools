"""Interactive 2D plotting of poses and pose sequences with plotly."""

from typing import Iterable, Mapping, Sequence

import numpy as np
import plotly.graph_objects as go

from sign_language_tools.pose.reference_frames import (
    SharedLandmark,
    align_reference_frames,
)
from sign_language_tools.pose.visualization.plotly.common import (
    AXIS_COLOR,
    Edge,
    LEGEND_BACKGROUND,
    MARGIN,
    PAPER_COLOR,
    SURFACE_COLOR,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    TITLE_MARGIN,
    TIMELINE_HEIGHT,
    check_edges,
    check_groups,
    frame_landmarks,
    frame_stride,
    per_group,
    playback_buttons,
    resolve_colors,
    select_frame,
    select_indices,
    timeline,
)


__all__ = [
    "plot_pose_2d",
    "plot_pose_sequence_2d",
]


_MIN_WIDTH = 240
_MAX_WIDTH = 2000


def plot_pose_2d(
    poses: Mapping[str, np.ndarray],
    edges: Mapping[str, Sequence[Edge]] | None = None,
    *,
    frame: int | None = None,
    align: Mapping[str, SharedLandmark] | None = None,
    colors: Mapping[str, str] | None = None,
    vertex_size: float | Mapping[str, float] = 6.0,
    edge_width: float | Mapping[str, float] = 2.0,
    show_indices: bool = False,
    show_axes: bool = False,
    title: str | None = None,
    x_lim: tuple[float, float] | None = None,
    y_lim: tuple[float, float] | None = None,
    refocus: bool = False,
    focus_pad: float = 0.02,
    height: int = 700,
    width: int | None = None,
    show: bool = True,
    renderer: str | None = None,
) -> go.Figure:
    """Plots a static pose as an interactive 2D skeleton.

    Every group in `poses` is drawn as a set of vertices, connected by the edges
    given under the same key in `edges`, and gets its own color and legend entry.
    Clicking a legend entry hides or shows both the vertices and the edges of
    that group. Landmarks that are `NaN` (undetected body parts) are skipped,
    along with the edges that touch them.

    Coordinates are assumed to follow the image convention: the origin is the
    top-left corner, `x` grows to the right and `y` grows downward. The y-axis is
    therefore drawn reversed, and both axes share the same scale so the pose
    keeps its proportions.

    Args:
        poses: Mapping from a group name (e.g. `"left_hand"`) to its landmarks,
            of shape `(L, C)` or, if `frame` is given, `(T, L, C)`, where `T` is
            the number of frames, `L` the number of landmarks and `C` the number
            of coordinates per landmark. Only the first two coordinates of each
            landmark are used.
        edges: Mapping from a group name to the edges drawn for that group, as
            `(start, end)` landmark index pairs. Keys must also be keys of
            `poses`; groups without an entry are drawn as vertices only.
        frame: Frame index to select from the groups of `poses` that are given as
            a `(T, L, C)` sequence. Required if any group is a sequence.
        align: Mapping from a group name to the
            [`SharedLandmark`][sign_language_tools.pose.reference_frames.SharedLandmark]
            that brings it into a common reference frame, applied after `frame`
            selection. Rarely needed here, since `x` and `y` usually already
            share a frame; see
            [`plot_pose_3d`][sign_language_tools.pose.visualization.plotly.graph_3d.plot_pose_3d],
            where the depth makes this essential.
        colors: Overrides the color of the given groups. Groups that are not
            listed keep their default color, see `DEFAULT_GROUP_COLORS`.
        vertex_size: Diameter of the vertex markers, in pixels, either for every
            group or per group. Use `0` to hide the vertices of a group.
        edge_width: Width of the edges, in pixels, either for every group or per
            group.
        show_indices: Whether to draw the index of each landmark next to its
            vertex. Mostly useful to build or debug edge definitions.
        show_axes: Whether to draw the ticks and labels of the x- and y-axis. The
            border around the data region is always drawn.
        title: Title of the figure. No title is drawn if `None`.
        x_lim: Override for the x-axis limits, as `(left, right)`.
        y_lim: Override for the y-axis limits, as `(top, bottom)`. Note the
            order: `y` grows downward.
        refocus: If `True`, compute the axis limits to fit tightly around the
            landmarks instead of using the default `[0, 1]` unit square. Ignored
            for an axis whose limits are given explicitly.
        focus_pad: Padding added around the landmarks when `refocus` is `True`.
            Ignored otherwise.
        height: Height of the figure, in pixels.
        width: Width of the figure, in pixels. Computed from `height` and the
            aspect ratio of the axis limits if `None`, so that the pose fills the
            figure instead of sitting in a band of empty background.
        show: Whether to display the figure immediately.
        renderer: Plotly renderer used when `show` is `True`, e.g. `"browser"` or
            `"notebook"`. Uses the plotly default if `None`.

    Returns:
        The plotly figure containing the pose.

    Raises:
        ValueError: If `poses` is empty, if `edges` contains a group that is
            missing from `poses`, if a group has an unexpected shape, if `frame`
            is missing for a group given as a sequence, or if an edge refers to a
            landmark that does not exist in its group.

    Example:
        >>> import numpy as np
        >>> from sign_language_tools.pose.visualization import plot_pose_2d
        >>> from sign_language_tools.pose.mediapipe.edges import HAND_EDGES, UPPER_POSE_EDGES
        >>> poses = {"pose": np.random.rand(33, 2), "right_hand": np.random.rand(21, 2)}
        >>> edges = {"pose": UPPER_POSE_EDGES, "right_hand": HAND_EDGES}
        >>> fig = plot_pose_2d(poses, edges, show=False)

        Plotting a single frame of a pose sequence, focused on the landmarks:

        >>> poses = {"right_hand": np.random.rand(100, 21, 3)}
        >>> fig = plot_pose_2d(poses, {"right_hand": HAND_EDGES}, frame=42, refocus=True, show=False)
    """
    edges = dict(edges or {})
    check_groups(poses, edges)

    landmarks = {name: select_frame(group, name, frame) for name, group in poses.items()}
    if align:
        landmarks = align_reference_frames(landmarks, align)
    landmarks = {name: group[:, :2] for name, group in landmarks.items()}
    for name, group_edges in edges.items():
        check_edges(group_edges, len(landmarks[name]), name)

    group_colors = resolve_colors(landmarks, colors)
    x_lim, y_lim = _resolve_limits(landmarks.values(), x_lim, y_lim, refocus, focus_pad)

    traces = _pose_traces(
        landmarks,
        edges,
        group_colors,
        vertex_size=vertex_size,
        edge_width=edge_width,
        show_indices=show_indices,
    )

    top_margin = TITLE_MARGIN if title else MARGIN
    figure = go.Figure(data=traces)
    figure.update_layout(
        title=dict(text=title, font=dict(color=TEXT_PRIMARY)) if title else None,
        paper_bgcolor=PAPER_COLOR,
        plot_bgcolor=SURFACE_COLOR,
        font=dict(color=TEXT_SECONDARY),
        legend=dict(
            font=dict(color=TEXT_SECONDARY),
            itemsizing="constant",
            xref="paper",
            yref="paper",
            x=0.01,
            y=0.99,
            xanchor="left",
            yanchor="top",
            bgcolor=LEGEND_BACKGROUND,
            borderwidth=0,
        ),
        margin=dict(b=MARGIN, l=MARGIN, r=MARGIN, t=top_margin),
        height=height,
        width=width if width is not None else _fitted_width(x_lim, y_lim, height, top_margin),
        hovermode="closest",
        xaxis=_axis_layout(x_lim, show_axes),
        yaxis=_axis_layout(y_lim, show_axes) | dict(scaleanchor="x", scaleratio=1),
    )

    if show:
        figure.show(renderer=renderer)
    return figure


def plot_pose_sequence_2d(
    poses: Mapping[str, np.ndarray],
    edges: Mapping[str, Sequence[Edge]] | None = None,
    *,
    frames: slice | Sequence[int] | None = None,
    max_frames: int | None = 500,
    fps: float = 25.0,
    speeds: Sequence[float] = (0.1, 0.25, 0.5, 1.0, 2.0),
    align: Mapping[str, SharedLandmark] | None = None,
    colors: Mapping[str, str] | None = None,
    vertex_size: float | Mapping[str, float] = 6.0,
    edge_width: float | Mapping[str, float] = 2.0,
    show_indices: bool = False,
    show_axes: bool = False,
    title: str | None = None,
    x_lim: tuple[float, float] | None = None,
    y_lim: tuple[float, float] | None = None,
    refocus: bool = False,
    focus_pad: float = 0.02,
    height: int = 700,
    width: int | None = None,
    show: bool = True,
    renderer: str | None = None,
) -> go.Figure:
    """Plots a pose sequence as an interactive 2D skeleton, with a timeline.

    Draws the same skeleton as
    [`plot_pose_2d`][sign_language_tools.pose.visualization.plotly.graph_2d.plot_pose_2d],
    with a play button and a slider to scrub through time. The axis limits are
    computed once over the whole selection, so the pose moves against a fixed
    frame instead of the axes rescaling under it.

    !!! warning "Every frame is embedded in the figure"

        A plotly animation carries a full copy of the data for each frame, at
        roughly 8 KB per frame for a body and two hands, and 30 KB per frame once
        the 478-landmark face is included. A few thousand frames therefore
        produce a figure of tens of megabytes, which is slow to build and slower
        to open.

        `max_frames` guards against this by keeping only every n-th frame and
        telling you the stride it used. To study a passage closely, select it
        with `frames` instead: a range shorter than `max_frames` is shown at full
        temporal resolution.

    Args:
        poses: Mapping from a group name (e.g. `"left_hand"`) to its pose
            sequence, of shape `(T, L, C)`, where `T` is the number of frames,
            `L` the number of landmarks and `C` the number of coordinates per
            landmark. Every group must have the same `T`. Only the first two
            coordinates of each landmark are used.
        edges: Mapping from a group name to the edges drawn for that group, as
            `(start, end)` landmark index pairs. Keys must also be keys of
            `poses`; groups without an entry are drawn as vertices only.
        frames: Frames to show, as a `slice` (e.g. `slice(100, 200)`) or an
            explicit sequence of indices. Shows the whole sequence if `None`.
        max_frames: Largest number of frames to embed. If the selection is
            longer, it is subsampled with a constant stride and a `UserWarning`
            reports it. Pass `None` to disable the guard and embed everything.
        fps: Rate that the `1×` button plays at, in frames of the **original**
            sequence per second. Set it to the frame rate of the source video and
            `1×` runs at real speed, whatever stride `max_frames` ended up using.
        speeds: Playback speeds offered as buttons, as multiples of `fps`. One
            button is drawn per speed, since plotly fixes the frame duration when
            playback starts and cannot change it while running: each button
            resumes from the current frame at its own rate.
        align: Mapping from a group name to the
            [`SharedLandmark`][sign_language_tools.pose.reference_frames.SharedLandmark]
            that brings it into a common reference frame, applied to each frame.
        colors: Overrides the color of the given groups. Groups that are not
            listed keep their default color, see `DEFAULT_GROUP_COLORS`.
        vertex_size: Diameter of the vertex markers, in pixels, either for every
            group or per group. Use `0` to hide the vertices of a group.
        edge_width: Width of the edges, in pixels, either for every group or per
            group.
        show_indices: Whether to draw the index of each landmark next to its
            vertex.
        show_axes: Whether to draw the ticks and labels of the x- and y-axis.
        title: Title of the figure. No title is drawn if `None`.
        x_lim: Override for the x-axis limits, as `(left, right)`.
        y_lim: Override for the y-axis limits, as `(top, bottom)`.
        refocus: If `True`, compute the axis limits to fit tightly around the
            landmarks of the whole selection, instead of using the default
            `[0, 1]` unit square.
        focus_pad: Padding added around the landmarks when `refocus` is `True`.
        height: Height of the figure, in pixels.
        width: Width of the figure, in pixels. Computed from `height` and the
            aspect ratio of the axis limits if `None`.
        show: Whether to display the figure immediately.
        renderer: Plotly renderer used when `show` is `True`. `"browser"` is the
            most reliable choice for an animation.

    Returns:
        The plotly figure, holding one plotly frame per shown frame.

    Raises:
        ValueError: If `poses` is empty, if a group is not a `(T, L, C)`
            sequence, if the groups have different lengths, if `frames` selects
            nothing, or if an edge refers to a landmark that does not exist.

    Example:
        >>> import numpy as np
        >>> from sign_language_tools.pose.visualization import plot_pose_sequence_2d
        >>> from sign_language_tools.pose.mediapipe.edges import HAND_EDGES, UPPER_POSE_EDGES
        >>> poses = {"pose": np.random.rand(300, 33, 3), "right_hand": np.random.rand(300, 21, 3)}
        >>> edges = {"pose": UPPER_POSE_EDGES, "right_hand": HAND_EDGES}
        >>> fig = plot_pose_sequence_2d(poses, edges, show=False)

        Studying one passage at full temporal resolution:

        >>> fig = plot_pose_sequence_2d(poses, edges, frames=slice(120, 180), show=False)
    """
    edges = dict(edges or {})
    check_groups(poses, edges)

    indices = select_indices(poses, frames, max_frames)
    sequence = [
        {name: group[:, :2] for name, group in frame_landmarks(poses, index, align).items()}
        for index in indices
    ]
    for name, group_edges in edges.items():
        check_edges(group_edges, len(sequence[0][name]), name)

    group_colors = resolve_colors(sequence[0], colors)
    x_lim, y_lim = _resolve_limits(
        [np.concatenate([frame[name] for frame in sequence]) for name in sequence[0]],
        x_lim,
        y_lim,
        refocus,
        focus_pad,
    )

    def traces_of(landmarks):
        return _pose_traces(
            landmarks,
            edges,
            group_colors,
            vertex_size=vertex_size,
            edge_width=edge_width,
            show_indices=show_indices,
        )

    top_margin = TITLE_MARGIN if title else MARGIN
    figure = go.Figure(
        data=traces_of(sequence[0]),
        frames=[
            go.Frame(data=traces_of(landmarks), name=str(index))
            for index, landmarks in zip(indices, sequence)
        ],
    )
    figure.update_layout(
        title=dict(text=title, font=dict(color=TEXT_PRIMARY)) if title else None,
        paper_bgcolor=PAPER_COLOR,
        plot_bgcolor=SURFACE_COLOR,
        font=dict(color=TEXT_SECONDARY),
        legend=dict(
            font=dict(color=TEXT_SECONDARY),
            itemsizing="constant",
            xref="paper",
            yref="paper",
            x=0.01,
            y=0.99,
            xanchor="left",
            yanchor="top",
            bgcolor=LEGEND_BACKGROUND,
            borderwidth=0,
        ),
        margin=dict(b=TIMELINE_HEIGHT, l=MARGIN, r=MARGIN, t=top_margin),
        height=height,
        width=(
            width
            if width is not None
            else _fitted_width(x_lim, y_lim, height, top_margin, TIMELINE_HEIGHT)
        ),
        hovermode="closest",
        xaxis=_axis_layout(x_lim, show_axes),
        yaxis=_axis_layout(y_lim, show_axes) | dict(scaleanchor="x", scaleratio=1),
        updatemenus=[playback_buttons(fps, frame_stride(indices), speeds)],
        sliders=[timeline(indices)],
    )

    if show:
        figure.show(renderer=renderer)
    return figure


# ---- Data preparation


def _resolve_limits(
    landmarks: Iterable[np.ndarray],
    x_lim: tuple[float, float] | None,
    y_lim: tuple[float, float] | None,
    refocus: bool,
    focus_pad: float,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Returns the `(left, right)` and `(top, bottom)` limits of both axes."""
    if not refocus:
        return x_lim or (0.0, 1.0), y_lim or (1.0, 0.0)

    points = np.concatenate([np.asarray(group).reshape(-1, 2) for group in landmarks])
    if np.all(np.isnan(points)):
        return x_lim or (0.0, 1.0), y_lim or (1.0, 0.0)

    x_min, y_min = np.nanmin(points, axis=0) - focus_pad
    x_max, y_max = np.nanmax(points, axis=0) + focus_pad
    return x_lim or (x_min, x_max), y_lim or (y_max, y_min)


# ---- Traces and layout


def _pose_traces(
    landmarks: Mapping[str, np.ndarray],
    edges: Mapping[str, Sequence[Edge]],
    group_colors: Mapping[str, str],
    *,
    vertex_size: float | Mapping[str, float],
    edge_width: float | Mapping[str, float],
    show_indices: bool,
) -> list[go.Scatter]:
    """Builds the edge and vertex trace of every group, in a stable order.

    The order matters for animation: a plotly frame replaces traces by position,
    so every frame of a sequence must lay its traces out identically.
    """
    traces = []
    for name, group in landmarks.items():
        traces.append(
            _edge_trace(
                group,
                edges.get(name, ()),
                name=name,
                color=group_colors[name],
                width=per_group(edge_width, name),
            )
        )
        traces.append(
            _vertex_trace(
                group,
                name=name,
                color=group_colors[name],
                size=per_group(vertex_size, name),
                show_indices=show_indices,
            )
        )
    return traces


def _vertex_trace(
    group: np.ndarray,
    *,
    name: str,
    color: str,
    size: float,
    show_indices: bool,
) -> go.Scatter:
    indices = np.arange(len(group))
    return go.Scatter(
        x=group[:, 0],
        y=group[:, 1],
        mode="markers+text" if show_indices else "markers",
        marker=dict(size=size, color=color),
        text=indices if show_indices else None,
        textposition="top center",
        textfont=dict(color=TEXT_SECONDARY, size=10),
        customdata=indices,
        hovertemplate="<b>%{fullData.name}</b> #%{customdata}<br>(%{x:.3f}, %{y:.3f})<extra></extra>",
        name=name,
        legendgroup=name,
        showlegend=True,
    )


def _edge_trace(
    group: np.ndarray,
    group_edges: Sequence[Edge],
    *,
    name: str,
    color: str,
    width: float,
) -> go.Scatter:
    """Draws every edge of a group as one trace, segments separated by `NaN` gaps."""
    segments = np.full((3 * len(group_edges), 2), np.nan)
    if group_edges:
        starts, ends = np.asarray(group_edges).T
        segments[0::3] = group[starts]
        segments[1::3] = group[ends]

    return go.Scatter(
        x=segments[:, 0],
        y=segments[:, 1],
        mode="lines",
        line=dict(width=width, color=color),
        hoverinfo="skip",
        name=name,
        legendgroup=name,
        showlegend=False,
    )


def _fitted_width(
    x_lim: tuple[float, float],
    y_lim: tuple[float, float],
    height: int,
    top_margin: int,
    bottom_margin: int = MARGIN,
) -> int:
    """Returns the width for which the plot area exactly fills the figure.

    Both axes share the same scale, so plotly shrinks the plot area until it
    matches the aspect ratio of the limits. Sizing the figure to that ratio up
    front leaves no empty band of background around the pose.
    """
    x_span = abs(x_lim[1] - x_lim[0])
    y_span = abs(y_lim[1] - y_lim[0])
    if x_span == 0 or y_span == 0:
        return height

    plot_height = max(height - top_margin - bottom_margin, 1)
    width = plot_height * (x_span / y_span) + 2 * MARGIN
    return int(round(min(max(width, _MIN_WIDTH), _MAX_WIDTH)))


def _axis_layout(limits: tuple[float, float], show_axes: bool) -> dict:
    """Draws a border around the data region, and the ticks only if asked."""
    return dict(
        range=limits,
        showgrid=False,
        zeroline=False,
        showline=True,
        linecolor=AXIS_COLOR,
        linewidth=1,
        mirror=True,
        showticklabels=show_axes,
        ticks="outside" if show_axes else "",
        constrain="domain",
        color=AXIS_COLOR,
    )
