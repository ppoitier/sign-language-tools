"""Drawing utility for 1-D time series.

Renders one or more 1-D signals as polylines on a sliding time window
centred on the current playback time `t`.  The layout deliberately
mirrors :mod:`annotations` so a `TimeSeriesComponent` and an
`AnnotationComponent` configured with the same ``t_lims`` align
horizontally.

Terminology:
  - **series**: a (T,) or (T, C) array of sample values.
  - **channel**: one of the C 1-D signals inside a multi-channel series.
"""
import cv2
import numpy as np

from sign_language_tools.player.drawing_utils.utils import to_frame_coords


# Default channel colours (BGR), cycled if there are more channels than colours.
_DEFAULT_COLORS: tuple[tuple[int, int, int], ...] = (
    (0, 255, 0),     # green
    (0, 165, 255),   # orange
    (255, 0, 0),     # blue
    (0, 0, 255),     # red
    (255, 255, 0),   # cyan
    (255, 0, 255),   # magenta
    (0, 255, 255),   # yellow
)


def draw_time_series(
    frame: np.ndarray,
    values: np.ndarray,
    t: float,
    fps: float,
    t_lims: np.ndarray,
    frame_lims: np.ndarray,
    y_lim: tuple[float, float] = (0.0, 1.0),
    channel_colors: list[tuple[int, int, int]] | None = None,
    channel_labels: list[str] | None = None,
    line_width: int = 1,
    ticks_color: tuple[int, int, int] = (255, 255, 255),
    background_color: tuple[int, int, int] | None = None,
    show_timeline: bool = True,
    show_legend: bool = True,
) -> np.ndarray:
    """Draw a (possibly multi-channel) 1-D time series onto *frame*.

    Args:
        frame: Target BGR image (modified in-place and returned).
        values: Either (T,) or (T, C) array of sample values.
        t: Current playback time in seconds (centres the view).
        fps: Sampling rate of *values* (samples per second).
        t_lims: (2, 2) array ``[[t_offset_min, t_offset_max], [_, _]]``
            defining the visible time window relative to *t*.  The
            second row is unused (y-range comes from ``y_lim``) but
            kept for layout symmetry with the annotation component.
        frame_lims: (2, 2) int32 array ``[[x_min, x_max], [y_min, y_max]]``
            defining the pixel region to draw into.
        y_lim: Value range mapped to the vertical extent.  Defaults to
            ``(0.0, 1.0)`` (probability-friendly).
        channel_colors: Per-channel BGR colours.  Cycled if shorter than
            the number of channels.  ``None`` uses the module default.
        channel_labels: Optional per-channel legend labels.
        line_width: Thickness of the polylines.
        ticks_color: Colour for the timeline bar and tick marks.
        background_color: If set, fill the region with this colour first.
        show_timeline: Whether to draw the timeline + t / t±1 ticks.
        show_legend: Whether to draw the channel legend (top-left).

    Returns:
        The modified *frame*.
    """
    # --- Normalise to (T, C) ---
    if values.ndim == 1:
        values = values[:, None]
    n_samples, n_channels = values.shape
    if n_samples < 2:
        return frame  # not enough to draw a line

    # --- Background ---
    if background_color is not None:
        cv2.rectangle(frame, tuple(frame_lims[:, 0]), tuple(frame_lims[:, 1]), background_color, -1)

    # --- Time window (centred on t), same convention as annotations ---
    view_lims = t_lims.copy()
    view_lims[0] += t
    # Override the y-range with the user's value range, so to_frame_coords
    # maps values directly to pixels.
    plot_lims = view_lims.copy()
    plot_lims[1] = np.array(y_lim, dtype=plot_lims.dtype)

    t_min, t_max = float(view_lims[0, 0]), float(view_lims[0, 1])

    # --- Slice to visible samples only (with a 1-sample margin so the line
    # extends to the edge instead of stopping inside the panel). ---
    idx_start = max(0, int(np.floor(t_min * fps)) - 1)
    idx_end = min(n_samples, int(np.ceil(t_max * fps)) + 2)
    if idx_end - idx_start < 2:
        # Nothing in view
        if show_timeline:
            _draw_timeline(frame, t, frame_lims, view_lims, ticks_color)
        return frame

    sample_indices = np.arange(idx_start, idx_end)
    sample_times = sample_indices.astype("float32") / fps

    # --- Resolve colours ---
    palette = channel_colors or list(_DEFAULT_COLORS)
    colors = [palette[i % len(palette)] for i in range(n_channels)]

    # --- Draw each channel ---
    visible = values[idx_start:idx_end]  # (M, C)
    for c in range(n_channels):
        pts_world = np.empty((visible.shape[0], 2), dtype="float32")
        pts_world[:, 0] = sample_times
        pts_world[:, 1] = visible[:, c]
        pts_px = to_frame_coords(pts_world, plot_lims, frame_lims)
        # cv2.polylines needs (1, M, 2) int32
        cv2.polylines(
            frame,
            [pts_px.reshape(-1, 1, 2)],
            isClosed=False,
            color=colors[c],
            thickness=line_width,
            lineType=cv2.LINE_AA,
        )

    # --- Timeline ---
    if show_timeline:
        _draw_timeline(frame, t, frame_lims, view_lims, ticks_color)

    # --- Legend ---
    if show_legend and channel_labels is not None:
        _draw_legend(frame, channel_labels, colors, frame_lims)

    return frame


# ======================================================================
# Internals
# ======================================================================


def _draw_timeline(
    frame: np.ndarray,
    t: float,
    frame_lims: np.ndarray,
    view_lims: np.ndarray,
    ticks_color: tuple[int, int, int],
) -> None:
    """Vertical cursor at t and faint marks at t-1 / t+1.

    Unlike the annotation timeline (which sits on the y=0.75 line of a
    normalised 0..1 area), the time-series cursor spans the full height
    of the drawing region.
    """
    y_top, y_bot = int(frame_lims[1, 0]), int(frame_lims[1, 1])

    # Map t, t-1, t+1 to pixel x coords through to_frame_coords
    pts_world = np.array(
        [[t, 0.0], [t - 1.0, 0.0], [t + 1.0, 0.0]],
        dtype="float32",
    )
    # We only care about the x-component, so any y-range works
    plot_lims = view_lims.copy()
    plot_lims[1] = np.array([0.0, 1.0], dtype=plot_lims.dtype)
    pts_px = to_frame_coords(pts_world, plot_lims, frame_lims)

    x_t, x_tm1, x_tp1 = int(pts_px[0, 0]), int(pts_px[1, 0]), int(pts_px[2, 0])

    # Current time: full-height cursor
    cv2.line(frame, (x_t, y_top), (x_t, y_bot), ticks_color, 1)
    # t-1 / t+1: short ticks at the bottom
    tick_h = max(4, (y_bot - y_top) // 12)
    cv2.line(frame, (x_tm1, y_bot - tick_h), (x_tm1, y_bot), ticks_color, 1)
    cv2.line(frame, (x_tp1, y_bot - tick_h), (x_tp1, y_bot), ticks_color, 1)


def _draw_legend(
    frame: np.ndarray,
    labels: list[str],
    colors: list[tuple[int, int, int]],
    frame_lims: np.ndarray,
    font: int = cv2.FONT_HERSHEY_SIMPLEX,
    font_scale: float = 0.4,
    thickness: int = 1,
    line_height: int = 14,
    swatch_w: int = 10,
    pad: int = 6,
) -> None:
    """Top-left legend with a colour swatch per labelled channel."""
    x0 = int(frame_lims[0, 0]) + pad
    y0 = int(frame_lims[1, 0]) + pad + line_height
    n = min(len(labels), len(colors))
    for i in range(n):
        y = y0 + i * line_height
        # swatch
        cv2.line(frame, (x0, y - line_height // 3), (x0 + swatch_w, y - line_height // 3),
                 colors[i], 2, cv2.LINE_AA)
        # label
        cv2.putText(
            frame, labels[i], (x0 + swatch_w + 4, y),
            font, fontScale=font_scale, color=colors[i],
            thickness=thickness, lineType=cv2.LINE_AA,
        )


# ======================================================================
# Standalone demo
# ======================================================================

if __name__ == "__main__":
    h, w, c = 600, 800, 3
    fps = 25.0
    duration_s = 10.0
    n = int(duration_s * fps)
    t_axis = np.arange(n, dtype="float32") / fps
    # Two-channel signal: a probability-like sigmoid pulse and a noisy cosine.
    sig1 = 1.0 / (1.0 + np.exp(-3.0 * (t_axis - 4.0)))
    sig2 = 0.5 + 0.4 * np.cos(2 * np.pi * 0.5 * t_axis) + 0.05 * np.random.randn(n)
    series = np.stack([sig1, sig2], axis=1).astype("float32")

    times = np.linspace(0, duration_s, 300, dtype="float32")
    for current_t in times:
        frame = np.zeros((h, w, c), dtype=np.uint8)
        draw_time_series(
            frame=frame,
            values=series,
            t=float(current_t),
            fps=fps,
            t_lims=np.array([[-4, 4], [0, 1]], dtype="float32"),
            frame_lims=np.array([[100, 700], [400, 550]]),
            y_lim=(0.0, 1.0),
            channel_labels=["sigmoid", "noisy cos"],
            background_color=(40, 40, 40),
        )
        cv2.imshow("time series", frame)
        cv2.waitKey(20)