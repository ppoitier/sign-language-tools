"""Shared building blocks for the plotly pose plots: colors, checks and the timeline."""

import warnings
from math import ceil
from typing import Mapping, Sequence

import numpy as np

from sign_language_tools.pose.reference_frames import SharedLandmark, align_reference_frames


__all__ = [
    "CATEGORICAL_PALETTE",
    "DEFAULT_GROUP_COLORS",
]


Edge = tuple[int, int]

CATEGORICAL_PALETTE = (
    "#3987e5",  # blue
    "#c98500",  # yellow
    "#d55181",  # magenta
    "#008300",  # green
    "#9085e9",  # violet
    "#d95926",  # orange
    "#199e70",  # aqua
    "#e66767",  # red
)
"""Fixed order of group colors, stepped for the dark plot surface.

Colors are assigned to groups in this order and never cycled arbitrarily, so a
group keeps its color when other groups are added or removed. The first four
slots are separable for viewers with color vision deficiency even when all four
are on screen at once; beyond four, groups remain identifiable through the
legend, the hover labels and their position in the frame."""

DEFAULT_GROUP_COLORS = {
    "pose": CATEGORICAL_PALETTE[0],
    "upper_pose": CATEGORICAL_PALETTE[0],
    "lower_pose": CATEGORICAL_PALETTE[0],
    "body": CATEGORICAL_PALETTE[0],
    "left_hand": CATEGORICAL_PALETTE[1],
    "right_hand": CATEGORICAL_PALETTE[2],
    "face": CATEGORICAL_PALETTE[3],
}
"""Color reserved for each of the usual landmark group names. Groups that are not
listed here take the remaining slots of `CATEGORICAL_PALETTE`, in order."""

SURFACE_COLOR = "#1a1a19"
PAPER_COLOR = "#111110"
TEXT_PRIMARY = "#ffffff"
TEXT_SECONDARY = "#c3c2b7"
AXIS_COLOR = "#4a4a47"
LEGEND_BACKGROUND = "rgba(17, 17, 16, 0.6)"

MARGIN = 8
TITLE_MARGIN = 48
BUTTON_ROW_HEIGHT = 42
TIMELINE_HEIGHT = 112


def select_frame(group: np.ndarray, name: str, frame: int | None) -> np.ndarray:
    """Returns the `(L, C)` landmarks of one group, for the requested frame."""
    group = np.asarray(group, dtype=float)

    if group.ndim == 3:
        if frame is None:
            raise ValueError(
                f"Group '{name}' is a sequence of shape {group.shape}:"
                " pass `frame` to choose the frame to plot."
            )
        group = group[frame]
    elif group.ndim != 2:
        raise ValueError(
            f"Group '{name}' has shape {group.shape}, expected `(L, C)` or `(T, L, C)`."
        )

    if group.shape[-1] < 2:
        raise ValueError(
            f"Group '{name}' has {group.shape[-1]} coordinate(s) per landmark, expected at least 2."
        )
    return group


def check_edges(group_edges: Sequence[Edge], landmark_count: int, name: str) -> None:
    """Raises if an edge of a group refers to a landmark outside of that group."""
    for start, end in group_edges:
        if start >= landmark_count or end >= landmark_count:
            raise ValueError(
                f"Edge ({start}, {end}) of group '{name}' refers to a landmark that does not"
                f" exist: the group only has {landmark_count} landmarks."
                " Are the edges of that group the right ones?"
            )


def check_groups(poses: Mapping[str, np.ndarray], edges: Mapping[str, Sequence[Edge]]) -> None:
    """Raises if there is nothing to plot, or if `edges` names an unknown group."""
    if not poses:
        raise ValueError("`poses` is empty: there is nothing to plot.")

    unknown_groups = set(edges) - set(poses)
    if unknown_groups:
        raise ValueError(
            f"`edges` contains groups that are missing from `poses`: {sorted(unknown_groups)}."
        )


def resolve_colors(
    landmarks: Mapping[str, np.ndarray],
    colors: Mapping[str, str] | None,
) -> dict[str, str]:
    """Gives a color to every group, from the overrides, the defaults, or the palette."""
    colors = colors or {}
    reserved = {DEFAULT_GROUP_COLORS[name] for name in landmarks if name in DEFAULT_GROUP_COLORS}
    available = [color for color in CATEGORICAL_PALETTE if color not in reserved]

    resolved = {}
    for index, name in enumerate(landmarks):
        if name in colors:
            resolved[name] = colors[name]
        elif name in DEFAULT_GROUP_COLORS:
            resolved[name] = DEFAULT_GROUP_COLORS[name]
        else:
            resolved[name] = (
                available[index % len(available)] if available else CATEGORICAL_PALETTE[0]
            )
    return resolved


def per_group(value: float | Mapping[str, float], name: str) -> float:
    """Returns the value set for a group, from either a shared value or a mapping."""
    return value[name] if isinstance(value, Mapping) else value


# ---- Sequences


def select_indices(
    poses: Mapping[str, np.ndarray],
    frames: slice | Sequence[int] | None,
    max_frames: int | None,
) -> np.ndarray:
    """Returns the frame indices to show, subsampled to fit within `max_frames`."""
    lengths = {}
    for name, group in poses.items():
        group = np.asarray(group)
        if group.ndim != 3:
            raise ValueError(
                f"Group '{name}' has shape {group.shape}, expected a `(T, L, C)` sequence."
                " Use the single-pose function to plot one frame."
            )
        lengths[name] = len(group)

    if len(set(lengths.values())) > 1:
        raise ValueError(f"The groups have different numbers of frames: {lengths}.")

    sequence_length = next(iter(lengths.values()))
    indices = np.arange(sequence_length)
    if frames is not None:
        indices = indices[frames] if isinstance(frames, slice) else indices[np.asarray(frames)]
    if len(indices) == 0:
        raise ValueError(f"`frames` selects no frame out of the {sequence_length} available.")

    if max_frames is not None and len(indices) > max_frames:
        stride = ceil(len(indices) / max_frames)
        warnings.warn(
            f"Showing {ceil(len(indices) / stride)} of {len(indices)} frames, one every"
            f" {stride}, to keep the figure small enough to open. Select a shorter passage"
            " with `frames` to see it at full temporal resolution, or raise `max_frames`.",
            UserWarning,
            stacklevel=3,
        )
        indices = indices[::stride]
    return indices


def frame_landmarks(
    poses: Mapping[str, np.ndarray],
    index: int,
    align: Mapping[str, SharedLandmark] | None,
) -> dict[str, np.ndarray]:
    """Returns the aligned `(L, C)` landmarks of every group, for one frame."""
    landmarks = {name: select_frame(group, name, index) for name, group in poses.items()}
    if align:
        landmarks = align_reference_frames(landmarks, align)
    return landmarks


def frame_stride(indices: np.ndarray) -> float:
    """Returns how many source frames separate two shown frames, typically.

    Playback speed is expressed against the original sequence, so a subsampled
    figure has to hold each of its frames on screen for longer to run at the same
    apparent rate.
    """
    if len(indices) < 2:
        return 1.0
    return float(np.median(np.diff(indices))) or 1.0


def playback_buttons(fps: float, stride: float, speeds: Sequence[float]) -> dict:
    """A pause button and one play button per speed.

    Plotly bakes the frame duration into the `animate` call, so speed cannot be
    changed while playing: each speed is its own play button, which resumes from
    the current frame at that rate.
    """
    pause = dict(frame=dict(duration=0, redraw=True), mode="immediate", transition=dict(duration=0))
    buttons = [dict(label="Pause", method="animate", args=[[None], pause])]

    for speed in speeds:
        play = dict(
            frame=dict(duration=1000.0 * stride / (fps * speed), redraw=True),
            fromcurrent=True,
            transition=dict(duration=0),
            mode="immediate",
        )
        buttons.append(dict(label=f"{speed:g}×", method="animate", args=[None, play]))

    return dict(
        type="buttons",
        direction="left",
        showactive=False,
        x=0,
        y=0,
        xanchor="left",
        yanchor="top",
        pad=dict(t=6, r=6),
        bgcolor=SURFACE_COLOR,
        bordercolor=AXIS_COLOR,
        borderwidth=1,
        font=dict(color=TEXT_SECONDARY, size=12),
        buttons=buttons,
    )


def timeline(indices: np.ndarray) -> dict:
    """A slider with one step per shown frame, labelled with the true frame index.

    The step labels are made transparent rather than dropped: with hundreds of
    steps they would collide into an unreadable band, but plotly reads the label
    of the active step to fill `currentvalue`, which is what actually tells you
    where you are.
    """
    step = dict(frame=dict(duration=0, redraw=True), mode="immediate", transition=dict(duration=0))
    return dict(
        active=0,
        x=0,
        len=1,
        y=0,
        xanchor="left",
        yanchor="top",
        pad=dict(t=BUTTON_ROW_HEIGHT),
        bgcolor=AXIS_COLOR,
        activebgcolor=TEXT_SECONDARY,
        bordercolor=AXIS_COLOR,
        borderwidth=1,
        tickcolor=AXIS_COLOR,
        ticklen=0,
        font=dict(color="rgba(0,0,0,0)", size=1),
        currentvalue=dict(
            prefix="Frame ",
            visible=True,
            xanchor="right",
            font=dict(color=TEXT_SECONDARY, size=12),
        ),
        steps=[
            dict(method="animate", label=str(index), args=[[str(index)], step])
            for index in indices
        ],
    )
