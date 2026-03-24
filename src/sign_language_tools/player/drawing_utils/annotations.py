"""Drawing utilities for temporal annotations (segment timelines).

Renders a horizontal timeline showing annotation spans (e.g. sign gloss
boundaries) relative to the current playback time.

Terminology:
  - **annotation**: a labelled time interval [start, end] in seconds.
  - **segment rectangle**: the on-screen rectangle representing an annotation.
  - **timeline**: the horizontal axis at the bottom of the annotation area.
"""
import cv2
import numpy as np

from sign_language_tools.player.drawing_utils.utils import to_frame_coords


# ======================================================================
# Public API
# ======================================================================


def draw_annotations(
    frame: np.ndarray,
    sorted_segments: np.ndarray,
    t: float,
    t_lims: np.ndarray,
    frame_lims: np.ndarray,
    labels: list[str] | None = None,
    ticks_color: tuple[int, int, int] = (255, 255, 255),
    background_color: tuple[int, int, int] | None = None,
    segment_color: tuple[int, int, int] = (0, 200, 0),
    text_color: tuple[int, int, int] = (255, 255, 255),
    filled: bool = False,
) -> np.ndarray:
    """Draw temporal annotation segments onto *frame*.

    Args:
        frame: Target BGR image (modified in-place and returned).
        sorted_segments: (N, 2) float32 array of [start, end] in seconds,
            sorted by start time.
        t: Current playback time in seconds (centres the view).
        t_lims: (2, 2) array ``[[t_offset_min, t_offset_max], [y_min, y_max]]``
            defining the visible time window (relative to *t*) and the
            normalised y-range.
        frame_lims: (2, 2) int32 array ``[[x_min, x_max], [y_min, y_max]]``
            defining the pixel region to draw into.
        labels: Optional list of N label strings (one per segment).
        ticks_color: Colour for the timeline and tick marks.
        background_color: If set, fill the region with this colour first.
        segment_color: Colour for the segment rectangles.
        text_color: Colour for the label text.
        filled: If True, draw filled rectangles; otherwise outlines.

    Returns:
        The modified *frame*.
    """
    # Shift the time window to be centred on `t`
    view_lims = t_lims.copy()
    view_lims[0] += t

    # Layout constants (in normalised y-coordinates)
    y_segment_top = 0.1
    y_timeline = 0.75

    # Clear background
    if background_color is not None:
        cv2.rectangle(frame, tuple(frame_lims[:, 0]), tuple(frame_lims[:, 1]), background_color, -1)

    # Pixel bounds of the drawing area
    x_px_min, x_px_max = int(frame_lims[0, 0]), int(frame_lims[0, 1])

    # Convert segment time-ranges to 2D rectangles in normalised coords,
    # then to pixel coords.
    rects_norm = _segments_to_rects(sorted_segments, y_segment_top, y_timeline)
    rects_px = to_frame_coords(rects_norm, view_lims, frame_lims)

    # Draw visible segments and collect label positions
    label_candidates: list[tuple[int, int, int, str]] = []  # (x_centre, x_start, x_end, label)

    for i, ((x_start, y_start), (x_end, y_end)) in enumerate(rects_px):
        # --- Visibility culling (fixes the premature-rectangle bug) ---
        if x_start > x_px_max:
            # All remaining segments are further right → stop
            break
        if x_end < x_px_min:
            # Entirely to the left of the visible area → skip
            continue

        # Clamp to visible region
        x_start_clamped = max(x_start, x_px_min)
        x_end_clamped = min(x_end, x_px_max)

        cv2.rectangle(
            frame,
            (x_start_clamped, y_start),
            (x_end_clamped, y_end),
            segment_color,
            thickness=-1 if filled else 1,
        )

        if labels is not None and i < len(labels):
            x_centre = x_start_clamped + (x_end_clamped - x_start_clamped) // 2
            label_candidates.append((x_centre, x_start_clamped, x_end_clamped, labels[i]))

    # Draw labels with overlap avoidance
    if label_candidates:
        y_top = int(rects_px[0, 0, 1]) if len(rects_px) > 0 else int(frame_lims[1, 0])
        _draw_labels(frame, label_candidates, y_top, text_color)

    # Timeline ticks
    _draw_timeline(frame, t, frame_lims, view_lims, y_timeline, ticks_color)

    return frame


# Keep the old name as an alias for backwards compatibility
draw_segments = draw_annotations


# ======================================================================
# Internals
# ======================================================================


def _segments_to_rects(
    segments: np.ndarray,
    y_top: float,
    y_bottom: float,
) -> np.ndarray:
    """Convert (N, 2) time-segments to (N, 2, 2) normalised rectangles.

    Each rectangle is ``[[x_start, y_top], [x_end, y_bottom]]``.
    """
    n = segments.shape[0]
    rects = np.empty((n, 2, 2), dtype="float32")
    rects[:, 0, 0] = segments[:, 0]  # x_start = segment start time
    rects[:, 0, 1] = y_top
    rects[:, 1, 0] = segments[:, 1]  # x_end   = segment end time
    rects[:, 1, 1] = y_bottom
    return rects


def _draw_labels(
    frame: np.ndarray,
    candidates: list[tuple[int, int, int, str]],
    y_top: int,
    text_color: tuple[int, int, int],
    font: int = cv2.FONT_HERSHEY_SIMPLEX,
    font_scale: float = 0.4,
    thickness: int = 1,
    line_height: int = 14,
    min_horizontal_gap: int = 10,
) -> None:
    """Place labels, stacking vertically only when they would overlap.

    Uses a greedy left-to-right sweep: each label is placed on the
    highest row where it doesn't collide horizontally with any label
    already on that row.
    """
    # rows[level] = list of (x_left, x_right) intervals already occupied
    rows: dict[int, list[tuple[int, int]]] = {}

    for x_centre, x_start, x_end, label in candidates:
        (text_w, text_h), _ = cv2.getTextSize(label, font, font_scale, thickness)

        # Centre the text horizontally within the segment, but clamp to segment bounds
        label_x = x_centre - text_w // 2
        label_x = max(label_x, x_start + 2)
        label_x_right = label_x + text_w

        # Find the first row level where this label fits without overlap
        level = 0
        while True:
            if level not in rows:
                break
            collision = any(
                not (label_x_right + min_horizontal_gap < existing_left
                     or label_x - min_horizontal_gap > existing_right)
                for existing_left, existing_right in rows[level]
            )
            if not collision:
                break
            level += 1

        # Record this label's horizontal extent on the chosen row
        rows.setdefault(level, []).append((label_x, label_x_right))

        label_y = y_top + (level + 1) * line_height
        cv2.putText(
            frame,
            label,
            (label_x, label_y),
            font,
            fontScale=font_scale,
            color=text_color,
            thickness=thickness,
            lineType=cv2.LINE_AA,
        )


def _draw_timeline(
    frame: np.ndarray,
    t: float,
    frame_lims: np.ndarray,
    view_lims: np.ndarray,
    y_timeline: float,
    ticks_color: tuple[int, int, int],
) -> None:
    """Draw the horizontal timeline with tick marks at t, t−1s, t+1s."""
    y_max_px = int(frame_lims[1, 1])

    tick_points = np.array(
        [
            # Horizontal timeline bar
            [view_lims[0, 0], y_timeline],
            [view_lims[0, 1], y_timeline],
            # Current time tick (tall)
            [t, 0.0],
            [t, y_timeline],
            # t − 1 s tick
            [t - 1, 0.5],
            [t - 1, y_timeline],
            # t + 1 s tick
            [t + 1, 0.5],
            [t + 1, y_timeline],
        ],
        dtype="float32",
    )
    pts = to_frame_coords(tick_points, view_lims, frame_lims)

    # Timeline bar
    cv2.line(frame, tuple(pts[0]), tuple(pts[1]), ticks_color, 3)
    # Current time
    cv2.line(frame, tuple(pts[2]), tuple(pts[3]), ticks_color, 1)
    # t − 1s
    cv2.line(frame, tuple(pts[4]), tuple(pts[5]), ticks_color, 1)
    # t + 1s
    cv2.line(frame, tuple(pts[6]), tuple(pts[7]), ticks_color, 1)

    # Tick labels
    _tick_label(frame, "t",    pts[3], y_max_px, ticks_color)
    _tick_label(frame, "t-1s", pts[5], y_max_px, ticks_color)
    _tick_label(frame, "t+1s", pts[7], y_max_px, ticks_color)


def _tick_label(
    frame: np.ndarray,
    text: str,
    anchor: np.ndarray,
    y_px: int,
    color: tuple[int, int, int],
    font: int = cv2.FONT_HERSHEY_SIMPLEX,
    font_scale: float = 0.5,
) -> None:
    """Draw a centred tick label below the timeline."""
    (text_w, _), _ = cv2.getTextSize(text, font, font_scale, 1)
    x = int(anchor[0]) - text_w // 2
    cv2.putText(
        frame, text, (x, y_px),
        font, fontScale=font_scale, color=color,
        thickness=1, lineType=cv2.LINE_AA,
    )


# ======================================================================
# Standalone demo
# ======================================================================

if __name__ == "__main__":
    h, w, c = 600, 800, 3
    times = np.linspace(0, 10, 300, dtype="float32")

    for current_t in times:
        frame = np.zeros((h, w, c), dtype=np.uint8)
        frame = draw_annotations(
            frame=frame,
            sorted_segments=np.array(
                [[0, 2], [1, 2], [3, 5]],
                dtype="float32",
            ),
            labels=["sign 1", "sign 2", "sign 3"],
            t=current_t,
            t_lims=np.array([[-4, 4], [0, 1]], dtype="float32"),
            frame_lims=np.array([[100, 700], [400, 550]]),
            background_color=(0, 0, 100),
        )
        cv2.imshow("frame", frame)
        cv2.waitKey(20)