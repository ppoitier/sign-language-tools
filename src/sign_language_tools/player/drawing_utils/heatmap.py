"""Drawing utility for 2-D heatmaps (e.g. mel spectrograms).

Renders a (T, F) array as a colour-mapped image on a sliding time
window centred on the current playback time `t`.  Layout mirrors
:mod:`annotations` and :mod:`time_series` so all three can sit
side-by-side and remain aligned in time.

Convention:
  - Input shape ``(T, F)`` — time-major, like the pose tensor.
  - Feature index 0 is drawn at the *bottom* of the panel (standard
    spectrogram orientation: low freq at the bottom).
"""
import cv2
import numpy as np

from sign_language_tools.player.drawing_utils.utils import to_frame_coords


def draw_heatmap(
    frame: np.ndarray,
    values: np.ndarray,
    t: float,
    fps: float,
    t_lims: np.ndarray,
    frame_lims: np.ndarray,
    v_lim: tuple[float, float] | None = None,
    colormap: int = cv2.COLORMAP_VIRIDIS,
    ticks_color: tuple[int, int, int] = (255, 255, 255),
    background_color: tuple[int, int, int] = (0, 0, 0),
    show_timeline: bool = True,
    show_freq_axis: bool = False,
    freq_lim: tuple[float, float] | None = None,
    freq_label: str = "",
) -> np.ndarray:
    """Draw a 2-D heatmap onto *frame*.

    Args:
        frame: Target BGR image (modified in-place and returned).
        values: (T, F) float array.
        t: Current playback time in seconds (centres the view).
        fps: Sampling rate of *values* along axis 0 (samples per second).
        t_lims: (2, 2) array ``[[t_offset_min, t_offset_max], [_, _]]``.
            Only the time row is used (the feature axis fills the panel
            vertically).
        frame_lims: (2, 2) int32 array ``[[x_min, x_max], [y_min, y_max]]``
            defining the pixel region to draw into.
        v_lim: ``(v_min, v_max)`` mapped to the colormap.  Defaults to
            the global min/max of *values* — stable across playback.
        colormap: OpenCV colormap constant (e.g. ``cv2.COLORMAP_VIRIDIS``).
        ticks_color: Colour for the time cursor and ticks.
        background_color: Region fill before drawing (covers out-of-range
            areas when the visible window extends past the data).
        show_timeline: Whether to draw the time cursor and ±1 s ticks.
        show_freq_axis: If True, write a small axis label on the left.
        freq_lim: Optional ``(f_min, f_max)`` used only for the axis
            label (purely informational — does not affect rendering).
        freq_label: Optional unit label (e.g. ``"mel"``, ``"Hz"``).

    Returns:
        The modified *frame*.
    """
    if values.ndim != 2:
        raise ValueError(f"draw_heatmap expects a 2-D array, got shape {values.shape}.")

    n_samples, n_features = values.shape
    if n_samples < 1 or n_features < 1:
        return frame

    # --- Background (covers everything; we paint the data on top) ---
    cv2.rectangle(
        frame, tuple(frame_lims[:, 0]), tuple(frame_lims[:, 1]),
        background_color, -1,
    )

    # --- Time window (same convention as annotations) ---
    view_lims = t_lims.copy()
    view_lims[0] += t
    t_min, t_max = float(view_lims[0, 0]), float(view_lims[0, 1])
    if t_max <= t_min:
        return frame

    # --- Pixel region ---
    x_px_min, x_px_max = int(frame_lims[0, 0]), int(frame_lims[0, 1])
    y_px_min, y_px_max = int(frame_lims[1, 0]), int(frame_lims[1, 1])
    panel_w = x_px_max - x_px_min
    panel_h = y_px_max - y_px_min
    if panel_w <= 0 or panel_h <= 0:
        return frame

    # --- Slice the visible columns (clamped to valid range) ---
    idx_start = max(0, int(np.floor(t_min * fps)))
    idx_end = min(n_samples, int(np.ceil(t_max * fps)) + 1)

    if idx_end > idx_start:
        visible = values[idx_start:idx_end]  # (M, F)

        # Normalise to [0, 255] using v_lim
        v_min, v_max = v_lim if v_lim is not None else (float(values.min()), float(values.max()))
        if v_max <= v_min:
            v_max = v_min + 1e-6
        norm = np.clip((visible - v_min) / (v_max - v_min), 0.0, 1.0)
        gray = (norm * 255).astype("uint8")  # shape (M, F)

        # We want time on X and features on Y, with feature 0 at the bottom.
        # gray is (time, feature); transposing gives (feature, time) which is
        # the correct (row, col) layout for an image with feature on Y.
        img = gray.T  # (F, M)
        img = np.flipud(img)  # feature 0 at the bottom row

        # Apply colormap → BGR
        colored = cv2.applyColorMap(img, colormap)  # (F, M, 3)

        # --- Figure out where the slice belongs on screen ---
        # Time-extent of the visible slice in *world* coordinates:
        slice_t_min = idx_start / fps
        slice_t_max = idx_end / fps  # exclusive end → matches resize span

        # Convert that span to pixel coords via to_frame_coords, using the
        # same plot_lims trick so the math stays identical to time_series.
        plot_lims = view_lims.copy()
        plot_lims[1] = np.array([0.0, 1.0], dtype=plot_lims.dtype)
        span_world = np.array(
            [[slice_t_min, 0.0], [slice_t_max, 1.0]], dtype="float32"
        )
        span_px = to_frame_coords(span_world, plot_lims, frame_lims)
        x0_target = int(span_px[0, 0])
        x1_target = int(span_px[1, 0])

        # Resize to the *unclipped* target size, then crop to the panel.
        full_w = x1_target - x0_target
        if full_w > 0:
            resized = cv2.resize(
                colored, (full_w, panel_h),
                interpolation=cv2.INTER_NEAREST,
            )
            # Crop the parts that fall outside the panel.
            left_crop = max(0, x_px_min - x0_target)
            right_crop = max(0, x1_target - x_px_max)
            cropped = resized[:, left_crop: full_w - right_crop]
            x0_paste = x0_target + left_crop
            if cropped.shape[1] > 0:
                frame[y_px_min:y_px_max, x0_paste:x0_paste + cropped.shape[1]] = cropped

    # --- Time cursor and side ticks ---
    if show_timeline:
        _draw_time_cursor(frame, t, frame_lims, view_lims, ticks_color)

    # --- Optional axis label ---
    if show_freq_axis:
        label = freq_label or ""
        if freq_lim is not None:
            cv2.putText(
                frame, f"{freq_lim[1]:g}", (x_px_min + 2, y_px_min + 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, ticks_color, 1, cv2.LINE_AA,
            )
            cv2.putText(
                frame, f"{freq_lim[0]:g}", (x_px_min + 2, y_px_max - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, ticks_color, 1, cv2.LINE_AA,
            )
        if label:
            cv2.putText(
                frame, label, (x_px_min + 2, (y_px_min + y_px_max) // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, ticks_color, 1, cv2.LINE_AA,
            )

    return frame


# ======================================================================
# Internals
# ======================================================================


def _draw_time_cursor(
    frame: np.ndarray,
    t: float,
    frame_lims: np.ndarray,
    view_lims: np.ndarray,
    ticks_color: tuple[int, int, int],
) -> None:
    """Full-height cursor at t plus short ticks at t±1 s."""
    y_top, y_bot = int(frame_lims[1, 0]), int(frame_lims[1, 1])

    plot_lims = view_lims.copy()
    plot_lims[1] = np.array([0.0, 1.0], dtype=plot_lims.dtype)
    pts_world = np.array(
        [[t, 0.0], [t - 1.0, 0.0], [t + 1.0, 0.0]],
        dtype="float32",
    )
    pts_px = to_frame_coords(pts_world, plot_lims, frame_lims)
    x_t, x_tm1, x_tp1 = int(pts_px[0, 0]), int(pts_px[1, 0]), int(pts_px[2, 0])

    cv2.line(frame, (x_t, y_top), (x_t, y_bot), ticks_color, 1)
    tick_h = max(4, (y_bot - y_top) // 12)
    cv2.line(frame, (x_tm1, y_bot - tick_h), (x_tm1, y_bot), ticks_color, 1)
    cv2.line(frame, (x_tp1, y_bot - tick_h), (x_tp1, y_bot), ticks_color, 1)


# ======================================================================
# Standalone demo
# ======================================================================

if __name__ == "__main__":
    h, w, c = 600, 800, 3
    fps = 50.0
    duration_s = 10.0
    n_t = int(duration_s * fps)
    n_f = 64

    # Synthetic "mel spectrogram": a slowly drifting band of energy.
    tt = np.arange(n_t, dtype="float32") / fps
    ff = np.arange(n_f, dtype="float32")
    centre = 20 + 30 * (0.5 + 0.5 * np.sin(2 * np.pi * 0.1 * tt))  # drifts in [20, 50]
    spec = np.exp(-((ff[None, :] - centre[:, None]) ** 2) / (2 * 5.0 ** 2))
    spec = (spec + 0.05 * np.random.rand(n_t, n_f)).astype("float32")

    times = np.linspace(0, duration_s, 300, dtype="float32")
    for current_t in times:
        frame = np.zeros((h, w, c), dtype=np.uint8)
        draw_heatmap(
            frame=frame,
            values=spec,
            t=float(current_t),
            fps=fps,
            t_lims=np.array([[-4, 4], [0, 1]], dtype="float32"),
            frame_lims=np.array([[100, 700], [400, 550]]),
            v_lim=(0.0, 1.0),
            show_freq_axis=True,
            freq_lim=(0.0, float(n_f)),
            freq_label="mel",
        )
        cv2.imshow("heatmap", frame)
        cv2.waitKey(20)