from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..backends import VideoBackend, TensorBackend, open_video
from .base import Component


@dataclass()
class VideoComponent(Component):
    """Renders frames from a video file using random-access decoding.

    Unlike the old sequential-read approach, `to_frame(t)` fetches the
    frame at time *t* directly.  This makes seeking instant: the player
    just changes the time counter and every component re-renders at the
    new position.
    """

    filepath: str = ""
    width: int = 0
    height: int = 0
    start_s: float = 0.0
    video_backend: str = "auto"  # "auto", "decord", "opencv"

    # Managed internally
    _backend: VideoBackend | None = field(default=None, repr=False, init=False)

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_file(
        cls,
        filepath: str,
        name: str | None = None,
        fps: float | None = None,
        speed: float = 1.0,
        start_ms: int | None = None,
        backend: str = "auto",
    ) -> "VideoComponent":
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Could not find video: {filepath}")

        vb = open_video(filepath, backend=backend)
        detected_fps = vb.fps
        width = vb.width
        height = vb.height
        vb.close()

        return cls(
            name=name or Path(filepath).stem,
            filepath=filepath,
            fps=fps or detected_fps,
            speed=speed,
            width=width,
            height=height,
            start_s=(start_ms / 1000.0) if start_ms is not None else 0.0,
            video_backend=backend,
        )

    @classmethod
    def from_tensor(
        cls,
        frames: np.ndarray,
        fps: float = 25.0,
        name: str | None = None,
        speed: float = 1.0,
        start_ms: int | None = None,
        is_rgb: bool = True,
    ) -> "VideoComponent":
        """Create a VideoComponent from an in-memory tensor or array.

        Args:
            frames: Array-like with shape ``(T, H, W, C)``.  Can be a
                numpy ndarray or a PyTorch tensor (``torch`` is imported
                only if needed).
            fps: Playback frame rate.
            name: Display name (auto-generated if None).
            speed: Playback speed multiplier.
            start_ms: Start offset in milliseconds.
            is_rgb: Whether channels are RGB (default, typical for
                PyTorch) or BGR (OpenCV convention).  RGB frames are
                converted to BGR on read.

        Example::

            # PyTorch tensor — no torch import needed at the call site
            comp = VideoComponent.from_tensor(model_output, fps=25)
            player.roots.append(comp)

            # Numpy array already in BGR
            comp = VideoComponent.from_tensor(frames_bgr, fps=30, is_rgb=False)
        """
        backend = TensorBackend(frames, fps=fps, is_rgb=is_rgb)
        comp = cls(
            name=name or cls._auto_name(),
            filepath="<tensor>",
            fps=fps,
            speed=speed,
            width=backend.width,
            height=backend.height,
            start_s=(start_ms / 1000.0) if start_ms is not None else 0.0,
            video_backend="tensor",
        )
        comp._backend = backend
        return comp

    @staticmethod
    def _auto_name() -> str:
        from uuid import uuid4
        return str(uuid4())

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def open(self) -> None:
        if self._backend is None:
            if self.video_backend == "tensor":
                raise RuntimeError(
                    "Tensor backend was not initialised — use from_tensor()."
                )
            self._backend = open_video(self.filepath, backend=self.video_backend)

    def close(self) -> None:
        if self._backend is not None:
            self._backend.close()
            self._backend = None

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def to_frame(self, t: float, parent_frame: np.ndarray | None = None) -> np.ndarray | None:
        """Return the video frame at time *t* (offset by start_s).

        Returns a blank frame when past the end of the video.
        """
        self.open()
        assert self._backend is not None
        frame = self._backend.read_frame_at_time(t + self.start_s)
        if frame is None:
            return self._blank_frame(self.width, self.height)
        return frame

    # ------------------------------------------------------------------
    # Info
    # ------------------------------------------------------------------

    @property
    def duration_s(self) -> float:
        self.open()
        assert self._backend is not None
        return self._backend.duration_s

    @property
    def frame_count(self) -> int:
        self.open()
        assert self._backend is not None
        return self._backend.frame_count
