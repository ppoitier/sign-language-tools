"""Additions to `overlays.py` — paste these alongside the existing
`AnnotationComponent` etc.

The two new components follow the same recipe as `AnnotationComponent`:
a dataclass holding the data + style, a `to_frame` that either uses the
parent's frame or makes a blank one, and a thin call into the matching
drawing utility.
"""
from dataclasses import dataclass, field

import cv2
import numpy as np

from sign_language_tools.player.drawing_utils.time_series import draw_time_series
from sign_language_tools.player.drawing_utils.heatmap import draw_heatmap

from .base import Component


@dataclass()
class TimeSeriesComponent(Component):
    """Draws a 1-D (or multi-channel 1-D) time series on a sliding window.

    The X-axis is time and is centred on the current playback time, using
    the same ``t_lims`` convention as :class:`AnnotationComponent`.
    The Y-axis is value, mapped from ``y_lim`` (defaults to ``(0, 1)``).
    """

    values: np.ndarray = field(default_factory=lambda: np.empty((0,), dtype="float32"))
    frame_lims: np.ndarray = field(default_factory=lambda: np.zeros((2, 2), dtype="int32"))
    t_lims: np.ndarray = field(default_factory=lambda: np.array([[-4, 4], [0, 1]], dtype="float32"))
    y_lim: tuple[float, float] = (0.0, 1.0)
    channel_labels: list[str] | None = None
    channel_colors: list[tuple[int, int, int]] | None = None
    line_width: int = 1
    ticks_color: tuple[int, int, int] = (255, 255, 255)
    background_color: tuple[int, int, int] | None = None
    show_timeline: bool = True
    show_legend: bool = True

    def to_frame(self, t: float, parent_frame: np.ndarray | None = None) -> np.ndarray:
        frame = parent_frame if parent_frame is not None else self._default_frame()
        draw_time_series(
            frame=frame,
            values=self.values,
            t=t,
            fps=self.fps,
            t_lims=self.t_lims,
            frame_lims=self.frame_lims,
            y_lim=self.y_lim,
            channel_colors=self.channel_colors,
            channel_labels=self.channel_labels,
            line_width=self.line_width,
            ticks_color=self.ticks_color,
            background_color=self.background_color,
            show_timeline=self.show_timeline,
            show_legend=self.show_legend,
        )
        return frame

    def _default_frame(self) -> np.ndarray:
        size = self.frame_lims[:, 1] - self.frame_lims[:, 0]
        return np.zeros((size[1], size[0], 3), dtype="uint8")


@dataclass()
class HeatmapComponent(Component):
    """Draws a 2-D array (e.g. mel spectrogram) on a sliding window.

    Input shape is ``(T, F)`` — time-major, like the pose tensor.
    Feature index 0 is rendered at the *bottom* of the panel (standard
    spectrogram orientation).
    """

    values: np.ndarray = field(default_factory=lambda: np.empty((0, 0), dtype="float32"))
    frame_lims: np.ndarray = field(default_factory=lambda: np.zeros((2, 2), dtype="int32"))
    t_lims: np.ndarray = field(default_factory=lambda: np.array([[-4, 4], [0, 1]], dtype="float32"))
    v_lim: tuple[float, float] | None = None  # None → auto (global min/max)
    colormap: int = cv2.COLORMAP_VIRIDIS
    ticks_color: tuple[int, int, int] = (255, 255, 255)
    background_color: tuple[int, int, int] = (0, 0, 0)
    show_timeline: bool = True
    show_freq_axis: bool = False
    freq_lim: tuple[float, float] | None = None
    freq_label: str = ""

    def to_frame(self, t: float, parent_frame: np.ndarray | None = None) -> np.ndarray:
        frame = parent_frame if parent_frame is not None else self._default_frame()
        draw_heatmap(
            frame=frame,
            values=self.values,
            t=t,
            fps=self.fps,
            t_lims=self.t_lims,
            frame_lims=self.frame_lims,
            v_lim=self.v_lim,
            colormap=self.colormap,
            ticks_color=self.ticks_color,
            background_color=self.background_color,
            show_timeline=self.show_timeline,
            show_freq_axis=self.show_freq_axis,
            freq_lim=self.freq_lim,
            freq_label=self.freq_label,
        )
        return frame

    def _default_frame(self) -> np.ndarray:
        size = self.frame_lims[:, 1] - self.frame_lims[:, 0]
        return np.zeros((size[1], size[0], 3), dtype="uint8")