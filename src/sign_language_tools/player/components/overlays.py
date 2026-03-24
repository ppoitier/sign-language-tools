from dataclasses import dataclass, field

import cv2
import numpy as np

from sign_language_tools.player.drawing_utils.poses import draw_pose
from sign_language_tools.player.drawing_utils.segments import draw_segments

from .base import Component


@dataclass()
class EmptyComponent(Component):
    """A solid-color rectangle.  Useful as a root component or spacer."""

    width: int = 300
    height: int = 200
    background_color: tuple[int, int, int] = (0, 0, 0)

    def to_frame(self, t: float, parent_frame: np.ndarray | None = None) -> np.ndarray:
        if parent_frame is not None:
            return parent_frame
        return self._blank_frame(self.width, self.height, self.background_color)


@dataclass()
class SkeletonComponent(Component):
    """Draws a 2-D pose skeleton onto the parent frame (or a blank one)."""

    poses: np.ndarray = field(default_factory=lambda: np.empty((0, 0, 0)))
    frame_lims: np.ndarray = field(default_factory=lambda: np.zeros((2, 2), dtype="int32"))
    vertex_lims: np.ndarray = field(default_factory=lambda: np.array([[0, 1], [0, 1]], dtype="float32"))
    edges: list[tuple[int, int]] | None = None
    vertex_color: tuple[int, int, int] = (255, 0, 0)
    edge_color: tuple[int, int, int] = (255, 255, 255)
    vertex_width: int = 1
    edge_width: int = 1

    def to_frame(self, t: float, parent_frame: np.ndarray | None = None) -> np.ndarray:
        frame = parent_frame if parent_frame is not None else self._default_frame()
        n_poses = self.poses.shape[0]
        idx = min(n_poses - 1, self.frame_index(t))
        draw_pose(
            frame=frame,
            pose=self.poses[idx, :, :2],
            edges=self.edges,
            vertex_lims=self.vertex_lims,
            frame_lims=self.frame_lims,
            vertex_color=self.vertex_color,
            edge_color=self.edge_color,
            vertex_width=self.vertex_width,
            edge_width=self.edge_width,
        )
        return frame

    def _default_frame(self) -> np.ndarray:
        size = self.frame_lims[:, 1] - self.frame_lims[:, 0]
        return np.zeros((size[1], size[0], 3), dtype="uint8")


@dataclass()
class AnnotationComponent(Component):
    """Draws temporal segment annotations (e.g. gloss boundaries)."""

    segments: np.ndarray = field(default_factory=lambda: np.empty((0, 2), dtype="float32"))
    labels: list[str] | None = None
    frame_lims: np.ndarray = field(default_factory=lambda: np.zeros((2, 2), dtype="int32"))
    t_lims: np.ndarray = field(default_factory=lambda: np.array([[-4, 4], [0, 1]], dtype="float32"))
    segment_color: tuple[int, int, int] = (0, 255, 0)
    text_color: tuple[int, int, int] = (255, 255, 255)
    ticks_color: tuple[int, int, int] = (255, 255, 255)
    background_color: tuple[int, int, int] | None = None
    filled: bool = False

    def to_frame(self, t: float, parent_frame: np.ndarray | None = None) -> np.ndarray:
        frame = parent_frame if parent_frame is not None else self._default_frame()
        draw_segments(
            frame=frame,
            sorted_segments=self.segments,
            t=t,
            t_lims=self.t_lims,
            frame_lims=self.frame_lims,
            labels=self.labels,
            segment_color=self.segment_color,
            text_color=self.text_color,
            background_color=self.background_color,
            filled=self.filled,
            ticks_color=self.ticks_color,
        )
        return frame

    def _default_frame(self) -> np.ndarray:
        size = self.frame_lims[:, 1] - self.frame_lims[:, 0]
        return np.zeros((size[1], size[0], 3), dtype="uint8")


@dataclass()
class PlaybackInfoComponent(EmptyComponent):
    """Renders a small text overlay showing FPS / frame / time."""

    text_scale: float = 0.5
    text_x: int = 10
    text_y: int = 20

    def to_frame(self, t: float, parent_frame: np.ndarray | None = None) -> np.ndarray:
        frame = super().to_frame(t, parent_frame)
        frame_nb = self.frame_index(t)
        cv2.putText(
            frame,
            f"FPS={self.fps} ; Frame={frame_nb} ; T={t:.2f}s",
            (self.text_x, self.text_y),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=self.text_scale,
            color=(255, 255, 255),
            thickness=1,
            lineType=cv2.LINE_AA,
        )
        return frame