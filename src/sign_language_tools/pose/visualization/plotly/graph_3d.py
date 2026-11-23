"""Interactive 3D plotting of poses and pose sequences with plotly."""

from typing import Mapping, Sequence

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
    TIMELINE_HEIGHT,
    TITLE_MARGIN,
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
    "plot_pose_3d",
    "plot_pose_sequence_3d",
]


_SCENE_PAD = 0.05


def plot_pose_3d(
    poses: Mapping[str, np.ndarray],
    edges: Mapping[str, Sequence[Edge]] | None = None,
    *,
    frame: int | None = None,
    align: Mapping[str, SharedLandmark] | None = None,
    colors: Mapping[str, str] | None = None,
    vertex_size: float | Mapping[str, float] = 3.0,
    edge_width: float | Mapping[str, float] = 3.0,
    show_indices: bool = False,
    show_axes: bool = False,
    title: str | None = None,
    height: int = 700,
    width: int | None = None,
    show: bool = True,
    renderer: str | None = None,
) -> go.Figure:
    """Plots a static pose as an interactive 3D skeleton.

    Behaves like
    [`plot_pose_2d`][sign_language_tools.pose.visualization.plotly.graph_2d.plot_pose_2d],
    in a scene that can be orbited and zoomed, with the third coordinate of each
    landmark used as depth.

    The landmarks are assumed to follow the image convention, where `y` grows
    downward and depth grows away from the camera. They are mapped to the scene
    so that the pose stands upright and is seen from the front: `x` to the right,
    depth into the screen, and `-y` upward. All three axes share one scale, so
    the pose keeps its proportions.

    !!! warning "Every group must share one reference frame"

        Unlike `x` and `y`, the depth of a landmark is usually measured from a
        different origin for each body part, and the groups cannot be drawn in a
        single scene until they are brought into a common frame. MediaPipe, for
        instance, measures the depth of the pose from the midpoint of the hips
        but the depth of each hand from that hand's own wrist, so hands plotted
        as-is sit at the depth of the hips instead of in front of the chest.

        Pass `align` to correct this.
        [`MEDIAPIPE_DEPTH_ALIGNMENT`][sign_language_tools.pose.mediapipe.reference_frames.MEDIAPIPE_DEPTH_ALIGNMENT]
        handles the output of
        [`extract_poses_from_video_file`][sign_language_tools.pose.mediapipe.extraction.extract_poses_from_video_file],
        and
        [`MEDIAPIPE_WORLD_ALIGNMENT`][sign_language_tools.pose.mediapipe.reference_frames.MEDIAPIPE_WORLD_ALIGNMENT]
        the metric `*_world_landmarks`. Nothing is aligned by default, since only
        you know which frame your landmarks are in.

    Args:
        poses: Mapping from a group name (e.g. `"left_hand"`) to its landmarks,
            of shape `(L, C)` or, if `frame` is given, `(T, L, C)`, where `T` is
            the number of frames, `L` the number of landmarks and `C` the number
            of coordinates per landmark. Groups with only two coordinates are
            drawn flat, at depth `0`.
        edges: Mapping from a group name to the edges drawn for that group, as
            `(start, end)` landmark index pairs. Keys must also be keys of
            `poses`; groups without an entry are drawn as vertices only.
        frame: Frame index to select from the groups of `poses` that are given as
            a `(T, L, C)` sequence. Required if any group is a sequence.
        align: Mapping from a group name to the
            [`SharedLandmark`][sign_language_tools.pose.reference_frames.SharedLandmark]
            that brings it into a common reference frame, applied after `frame`
            selection. See the warning above.
        colors: Overrides the color of the given groups. Groups that are not
            listed keep their default color, see `DEFAULT_GROUP_COLORS`.
        vertex_size: Diameter of the vertex markers, in pixels, either for every
            group or per group. Use `0` to hide the vertices of a group.
        edge_width: Width of the edges, in pixels, either for every group or per
            group.
        show_indices: Whether to draw the index of each landmark next to its
            vertex. Mostly useful to build or debug edge definitions.
        show_axes: Whether to draw the three axes, with their ticks and labels.
        title: Title of the figure. No title is drawn if `None`.
        height: Height of the figure, in pixels.
        width: Width of the figure, in pixels. Fits the container if `None`.
        show: Whether to display the figure immediately.
        renderer: Plotly renderer used when `show` is `True`, e.g. `"browser"` or
            `"notebook"`. Uses the plotly default if `None`.

    Returns:
        The plotly figure containing the pose.

    Raises:
        ValueError: If `poses` is empty, if `edges` contains a group that is
            missing from `poses`, if a group has an unexpected shape, if `frame`
            is missing for a group given as a sequence, or if an edge or the
            alignment refers to a landmark that does not exist.

    Example:
        >>> import numpy as np
        >>> from sign_language_tools.pose.visualization import plot_pose_3d
        >>> from sign_language_tools.pose.mediapipe.reference_frames import (
        ...     MEDIAPIPE_DEPTH_ALIGNMENT,
        ... )
        >>> from sign_language_tools.pose.mediapipe.edges import HAND_EDGES, UPPER_POSE_EDGES
        >>> poses = {"pose": np.random.rand(33, 3), "right_hand": np.random.rand(21, 3)}
        >>> edges = {"pose": UPPER_POSE_EDGES, "right_hand": HAND_EDGES}
        >>> fig = plot_pose_3d(poses, edges, align=MEDIAPIPE_DEPTH_ALIGNMENT, show=False)
    """
    edges = dict(edges or {})
    check_groups(poses, edges)

    landmarks = {name: select_frame(group, name, frame) for name, group in poses.items()}
    if align:
        landmarks = align_reference_frames(landmarks, align)
    for name, group_edges in edges.items():
        check_edges(group_edges, len(landmarks[name]), name)

    group_colors = resolve_colors(landmarks, colors)

    traces = _pose_traces(
        {name: _to_scene(group) for name, group in landmarks.items()},
        edges,
        group_colors,
        vertex_size=vertex_size,
        edge_width=edge_width,
        show_indices=show_indices,
    )

    figure = go.Figure(data=traces)
    figure.update_layout(
        title=dict(text=title, font=dict(color=TEXT_PRIMARY)) if title else None,
        paper_bgcolor=PAPER_COLOR,
        font=dict(color=TEXT_SECONDARY),
        legend=_legend_layout(),
        margin=dict(b=MARGIN, l=MARGIN, r=MARGIN, t=TITLE_MARGIN if title else MARGIN),
        height=height,
        width=width,
        scene=_scene_layout(show_axes),
    )

    if show:
        figure.show(renderer=renderer)
    return figure


def plot_pose_sequence_3d(
    poses: Mapping[str, np.ndarray],
    edges: Mapping[str, Sequence[Edge]] | None = None,
    *,
    frames: slice | Sequence[int] | None = None,
    max_frames: int | None = 500,
    fps: float = 25.0,
    speeds: Sequence[float] = (0.1, 0.25, 0.5, 1.0, 2.0),
    align: Mapping[str, SharedLandmark] | None = None,
    colors: Mapping[str, str] | None = None,
    vertex_size: float | Mapping[str, float] = 3.0,
    edge_width: float | Mapping[str, float] = 3.0,
    show_indices: bool = False,
    show_axes: bool = False,
    title: str | None = None,
    height: int = 700,
    width: int | None = None,
    show: bool = True,
    renderer: str | None = None,
) -> go.Figure:
    """Plots a pose sequence as an interactive 3D skeleton, with a timeline.

    The 3D counterpart of
    [`plot_pose_sequence_2d`][sign_language_tools.pose.visualization.plotly.graph_2d.plot_pose_sequence_2d],
    with the same playback controls. Two things are handled for you so that the
    scene stays usable while it plays:

    - **The scene keeps its bounds.** The axis ranges are computed once over the
      whole selection, so the pose moves inside a fixed box instead of the scene
      rescaling around it on every frame.
    - **The camera survives playback.** Orbiting or zooming is preserved as the
      frames advance, so you can pick a viewpoint and then watch the sign from
      it. Double-click the scene to return to the default front view.

    !!! warning "Reference frames, and figure size"

        The depth of each group is measured from a different origin, exactly as
        for a single pose: pass `align` to correct it, see
        [`plot_pose_3d`][sign_language_tools.pose.visualization.plotly.graph_3d.plot_pose_3d].

        Every frame is also embedded in the figure, so `max_frames` caps the
        number kept, and `frames` selects a passage to study at full temporal
        resolution.

    Args:
        poses: Mapping from a group name (e.g. `"left_hand"`) to its pose
            sequence, of shape `(T, L, C)`, where `T` is the number of frames,
            `L` the number of landmarks and `C` the number of coordinates per
            landmark. Every group must have the same `T`. Groups with only two
            coordinates are drawn flat, at depth `0`.
        edges: Mapping from a group name to the edges drawn for that group, as
            `(start, end)` landmark index pairs. Keys must also be keys of
            `poses`; groups without an entry are drawn as vertices only.
        frames: Frames to show, as a `slice` (e.g. `slice(100, 200)`) or an
            explicit sequence of indices. Shows the whole sequence if `None`.
        max_frames: Largest number of frames to embed. If the selection is
            longer, it is subsampled with a constant stride and a `UserWarning`
            reports it. Pass `None` to disable the guard.
        fps: Rate that the `1×` button plays at, in frames of the **original**
            sequence per second, so `1×` runs at real speed whatever stride
            `max_frames` ended up using.
        speeds: Playback speeds offered as buttons, as multiples of `fps`.
        align: Mapping from a group name to the
            [`SharedLandmark`][sign_language_tools.pose.reference_frames.SharedLandmark]
            that brings it into a common reference frame, applied to each frame.
        colors: Overrides the color of the given groups.
        vertex_size: Diameter of the vertex markers, in pixels, either for every
            group or per group. Use `0` to hide the vertices of a group.
        edge_width: Width of the edges, in pixels, either for every group or per
            group.
        show_indices: Whether to draw the index of each landmark next to its
            vertex.
        show_axes: Whether to draw the three axes, with their ticks and labels.
        title: Title of the figure. No title is drawn if `None`.
        height: Height of the figure, in pixels.
        width: Width of the figure, in pixels. Fits the container if `None`.
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
        >>> from sign_language_tools.pose.visualization import plot_pose_sequence_3d
        >>> from sign_language_tools.pose.mediapipe.reference_frames import (
        ...     MEDIAPIPE_DEPTH_ALIGNMENT,
        ... )
        >>> from sign_language_tools.pose.mediapipe.edges import HAND_EDGES, UPPER_POSE_EDGES
        >>> poses = {"pose": np.random.rand(300, 33, 3), "right_hand": np.random.rand(300, 21, 3)}
        >>> edges = {"pose": UPPER_POSE_EDGES, "right_hand": HAND_EDGES}
        >>> fig = plot_pose_sequence_3d(poses, edges, align=MEDIAPIPE_DEPTH_ALIGNMENT, show=False)
    """
    edges = dict(edges or {})
    check_groups(poses, edges)

    indices = select_indices(poses, frames, max_frames)
    sequence = [
        {name: _to_scene(group) for name, group in frame_landmarks(poses, index, align).items()}
        for index in indices
    ]
    for name, group_edges in edges.items():
        check_edges(group_edges, len(sequence[0][name]), name)

    group_colors = resolve_colors(sequence[0], colors)
    ranges = _scene_ranges(sequence)

    def traces_of(scene_points):
        return _pose_traces(
            scene_points,
            edges,
            group_colors,
            vertex_size=vertex_size,
            edge_width=edge_width,
            show_indices=show_indices,
        )

    figure = go.Figure(
        data=traces_of(sequence[0]),
        frames=[
            go.Frame(data=traces_of(scene_points), name=str(index))
            for index, scene_points in zip(indices, sequence)
        ],
    )
    figure.update_layout(
        title=dict(text=title, font=dict(color=TEXT_PRIMARY)) if title else None,
        paper_bgcolor=PAPER_COLOR,
        font=dict(color=TEXT_SECONDARY),
        legend=_legend_layout(),
        margin=dict(
            b=TIMELINE_HEIGHT, l=MARGIN, r=MARGIN, t=TITLE_MARGIN if title else MARGIN
        ),
        height=height,
        width=width,
        # Without a stable `uirevision`, plotly rebuilds the scene on every frame
        # and snaps the camera back to its default, undoing any orbit the moment
        # playback starts.
        uirevision="pose-sequence",
        scene=_scene_layout(show_axes, ranges),
        updatemenus=[playback_buttons(fps, frame_stride(indices), speeds)],
        sliders=[timeline(indices)],
    )

    if show:
        figure.show(renderer=renderer)
    return figure


# ---- Coordinates


def _to_scene(group: np.ndarray) -> np.ndarray:
    """Maps image-convention landmarks to scene coordinates, as an `(L, 3)` array.

    The scene is right-handed with the vertical axis last, so the pose stands
    upright: `x` stays the horizontal axis, the depth becomes the second axis,
    and `-y` becomes the vertical one because image coordinates grow downward.
    """
    depth = group[:, 2] if group.shape[-1] > 2 else np.zeros(len(group))
    return np.stack([group[:, 0], depth, -group[:, 1]], axis=-1)


# ---- Traces and layout


def _pose_traces(
    scene_points: Mapping[str, np.ndarray],
    edges: Mapping[str, Sequence[Edge]],
    group_colors: Mapping[str, str],
    *,
    vertex_size: float | Mapping[str, float],
    edge_width: float | Mapping[str, float],
    show_indices: bool,
) -> list[go.Scatter3d]:
    """Builds the edge and vertex trace of every group, in a stable order.

    The order matters for animation: a plotly frame replaces traces by position,
    so every frame of a sequence must lay its traces out identically.
    """
    traces = []
    for name, group in scene_points.items():
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
    scene_points: np.ndarray,
    *,
    name: str,
    color: str,
    size: float,
    show_indices: bool,
) -> go.Scatter3d:
    indices = np.arange(len(scene_points))
    return go.Scatter3d(
        x=scene_points[:, 0],
        y=scene_points[:, 1],
        z=scene_points[:, 2],
        mode="markers+text" if show_indices else "markers",
        marker=dict(size=size, color=color),
        text=indices if show_indices else None,
        textposition="top center",
        textfont=dict(color=TEXT_SECONDARY, size=10),
        customdata=indices,
        hovertemplate=(
            "<b>%{fullData.name}</b> #%{customdata}"
            "<br>x %{x:.3f} · y %{z:.3f} · depth %{y:.3f}<extra></extra>"
        ),
        name=name,
        legendgroup=name,
        showlegend=True,
    )


def _edge_trace(
    scene_points: np.ndarray,
    group_edges: Sequence[Edge],
    *,
    name: str,
    color: str,
    width: float,
) -> go.Scatter3d:
    """Draws every edge of a group as one trace, segments separated by `NaN` gaps."""
    segments = np.full((3 * len(group_edges), 3), np.nan)
    if group_edges:
        starts, ends = np.asarray(group_edges).T
        segments[0::3] = scene_points[starts]
        segments[1::3] = scene_points[ends]

    return go.Scatter3d(
        x=segments[:, 0],
        y=segments[:, 1],
        z=segments[:, 2],
        mode="lines",
        line=dict(width=width, color=color),
        hoverinfo="skip",
        name=name,
        legendgroup=name,
        showlegend=False,
    )


def _scene_axis(
    title: str,
    show_axes: bool,
    axis_range: tuple[float, float] | None = None,
) -> dict:
    return dict(
        title=dict(text=title if show_axes else ""),
        visible=show_axes,
        showbackground=False,
        showgrid=show_axes,
        zeroline=False,
        gridcolor=AXIS_COLOR,
        color=AXIS_COLOR,
        range=axis_range,
    )


def _scene_layout(
    show_axes: bool,
    ranges: tuple[tuple[float, float], ...] | None = None,
) -> dict:
    """The scene, seen from the front with the pose upright.

    `aspectmode="data"` derives the box proportions from the axis ranges, so
    fixing the ranges also fixes the aspect: the pose neither stretches nor
    rescales as the sequence plays.
    """
    x_range, depth_range, y_range = ranges or (None, None, None)
    return dict(
        xaxis=_scene_axis("x", show_axes, x_range),
        yaxis=_scene_axis("depth", show_axes, depth_range),
        zaxis=_scene_axis("y", show_axes, y_range),
        aspectmode="data",
        bgcolor=SURFACE_COLOR,
        camera=dict(
            eye=dict(x=0.0, y=-2.0, z=0.0),
            up=dict(x=0.0, y=0.0, z=1.0),
            center=dict(x=0.0, y=0.0, z=0.0),
        ),
    )


def _legend_layout() -> dict:
    return dict(
        font=dict(color=TEXT_SECONDARY),
        itemsizing="constant",
        x=0.01,
        y=0.99,
        xanchor="left",
        yanchor="top",
        bgcolor=LEGEND_BACKGROUND,
        borderwidth=0,
    )


def _scene_ranges(
    sequence: Sequence[Mapping[str, np.ndarray]],
) -> tuple[tuple[float, float], ...]:
    """Returns the `(low, high)` bounds of each scene axis, over the whole sequence.

    Computing these once is what keeps the pose from being rescaled frame by
    frame. Frames where a body part is missing hold `NaN` and are ignored.
    """
    points = np.concatenate(
        [group for scene_points in sequence for group in scene_points.values()]
    )
    if np.all(np.isnan(points)):
        return ((-1.0, 1.0),) * 3

    low = np.nanmin(points, axis=0)
    high = np.nanmax(points, axis=0)
    pad = np.where(high > low, (high - low) * _SCENE_PAD, 1.0)
    return tuple(zip(low - pad, high + pad))
