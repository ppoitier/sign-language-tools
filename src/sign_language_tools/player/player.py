from dataclasses import dataclass, field
from time import time

import cv2
import numpy as np

from .components import (
    AnnotationComponent,
    Component,
    EmptyComponent,
    PlaybackInfoComponent,
    SkeletonComponent,
    VideoComponent,
)
from .rendering import display_frame, render_tree
from .utils import segments_to_seconds


@dataclass
class PlaybackClock:
    """Tracks the current playback time.

    The clock advances in real-time (scaled by *speed*) while unpaused,
    and can be jumped to any position via `seek()`.
    """

    speed: float = 1.0
    paused: bool = False
    _current_t: float = 0.0
    _last_wall: float = field(default_factory=time)

    @property
    def t(self) -> float:
        """Current playback time in seconds."""
        if not self.paused:
            now = time()
            self._current_t += (now - self._last_wall) * self.speed
            self._last_wall = now
        return self._current_t

    def seek(self, t: float) -> None:
        """Jump to an absolute time (clamped to >= 0)."""
        self._current_t = max(0.0, t)
        self._last_wall = time()

    def seek_delta(self, delta_s: float) -> None:
        """Jump forward/backward by *delta_s* seconds."""
        self.seek(self._current_t + delta_s)

    def toggle_pause(self) -> None:
        if self.paused:
            # Resuming — reset wall clock so we don't jump
            self._last_wall = time()
        self.paused = not self.paused


# OpenCV key codes (Linux — may differ on macOS/Windows)
_KEY_Q = ord("q")
_KEY_SPACE = ord(" ")
_KEYS_LEFT = {
    65361,    # Linux / GTK
    2424832,  # Windows
    63234,    # macOS
}
_KEYS_RIGHT = {
    65363,    # Linux / GTK
    2555904,  # Windows
    63235,    # macOS
}

_SEEK_STEP_S = 10.0  # seconds per arrow press


class VideoPlayer:
    """Modular video player for sign-language visualisation.

    Components are arranged in a forest (list of trees).  Root-level
    components each get their own OpenCV window; children render into
    their parent's frame.

    **Seeking model** — the player maintains a single global clock.
    Every component receives the current time `t` and renders the frame
    at that instant.  Because `VideoComponent` now uses random-access
    decoding (via decord or OpenCV `set(POS_FRAMES)`), seeking is just
    changing `t` — no stream reset needed.
    """

    def __init__(self, default_fps: float = 24, default_size: tuple[int, int] = (800, 600)):
        self.roots: list[Component] = []
        self.default_fps = default_fps
        self.default_size = default_size  # (width, height)
        self._clock: PlaybackClock | None = None

    def _find(self, name: str) -> Component:
        for root in self.roots:
            found = self._search(root, name)
            if found is not None:
                return found
        raise ValueError(f"Component '{name}' not found.")

    @staticmethod
    def _search(node: Component, name: str) -> Component | None:
        if node.name == name:
            return node
        for child in node.children:
            found = VideoPlayer._search(child, name)
            if found is not None:
                return found
        return None

    def _attach(self, component: Component, parent_name: str | None) -> None:
        if parent_name is None:
            self.roots.append(component)
        else:
            self._find(parent_name).add_child(component)

    def attach_video_file(
        self,
        filepath: str,
        name: str | None = None,
        fps: float | None = None,
        speed: float = 1.0,
        start_ms: int | None = None,
        backend: str = "auto",
    ) -> VideoComponent:
        comp = VideoComponent.from_file(
            filepath, name=name, fps=fps, speed=speed,
            start_ms=start_ms, backend=backend,
        )
        self.default_fps = comp.fps
        self.default_size = (comp.width, comp.height)
        self.roots.append(comp)
        return comp

    def attach_video_tensor(
        self,
        frames: np.ndarray,
        fps: float = 25.0,
        name: str | None = None,
        speed: float = 1.0,
        start_ms: int | None = None,
        is_rgb: bool = True,
    ) -> VideoComponent:
        """Attach an in-memory video tensor (numpy array or PyTorch tensor).

        Args:
            frames: Array-like with shape ``(T, H, W, C)``.
            fps: Playback frame rate.
            name: Display name.
            speed: Speed multiplier.
            start_ms: Start offset in ms.
            is_rgb: True if channels are RGB (PyTorch default), False for BGR.

        Example::

            player.attach_tensor(model_output, fps=25, name="generated")
        """
        comp = VideoComponent.from_tensor(
            frames,
            fps=fps,
            name=name,
            speed=speed,
            start_ms=start_ms,
            is_rgb=is_rgb,
        )
        self.default_fps = comp.fps
        self.default_size = (comp.width, comp.height)
        self.roots.append(comp)
        return comp

    def attach_empty(
        self,
        width: int | None = None,
        height: int | None = None,
        name: str | None = None,
        parent_name: str | None = None,
        fps: float | None = None,
        background_color: tuple[int, int, int] = (0, 0, 0),
    ) -> EmptyComponent:
        comp = EmptyComponent(
            name=name or self._auto_name(),
            fps=fps or self.default_fps,
            width=width or self.default_size[0],
            height=height or self.default_size[1],
            background_color=background_color,
        )
        self._attach(comp, parent_name)
        return comp

    def attach_poses(
        self,
        pose_seq: np.ndarray,
        edges: list[tuple[int, int]] | None = None,
        name: str | None = None,
        parent_name: str | None = None,
        fps: float | None = None,
        speed: float = 1.0,
        x_lim: tuple[int, int] | None = None,
        y_lim: tuple[int, int] | None = None,
        vertex_x_lim: tuple[float, float] = (0.0, 1.0),
        vertex_y_lim: tuple[float, float] = (0.0, 1.0),
        vertex_color: tuple[int, int, int] = (255, 0, 0),
        edge_color: tuple[int, int, int] = (255, 255, 255),
        vertex_width: int = 1,
        edge_width: int = 1,
    ) -> SkeletonComponent:
        x_lim = x_lim or (0, self.default_size[0])
        y_lim = y_lim or (0, self.default_size[1])
        comp = SkeletonComponent(
            name=name or self._auto_name(),
            fps=fps or self.default_fps,
            speed=speed,
            poses=pose_seq,
            edges=edges,
            frame_lims=np.array([x_lim, y_lim], dtype="int32"),
            vertex_lims=np.array([vertex_x_lim, vertex_y_lim], dtype="float32"),
            vertex_color=vertex_color,
            edge_color=edge_color,
            vertex_width=vertex_width,
            edge_width=edge_width,
        )
        self._attach(comp, parent_name)
        return comp

    def attach_segments(
        self,
        segments: np.ndarray,
        unit: str = "s",
        labels: list[str] | None = None,
        name: str | None = None,
        parent_name: str | None = None,
        fps: float | None = None,
        speed: float = 1.0,
        x_lim: tuple[int, int] = (0, 300),
        y_lim: tuple[int, int] = (0, 200),
        segment_color: tuple[int, int, int] = (0, 255, 0),
        text_color: tuple[int, int, int] = (255, 255, 255),
        background_color: tuple[int, int, int] | None = None,
        ticks_color: tuple[int, int, int] = (255, 255, 255),
        filled: bool = False,
    ) -> AnnotationComponent:
        effective_fps = fps or self.default_fps
        comp = AnnotationComponent(
            name=name or self._auto_name(),
            fps=effective_fps,
            speed=speed,
            segments=segments_to_seconds(segments, unit, effective_fps),
            labels=labels,
            frame_lims=np.array([x_lim, y_lim], dtype="int32"),
            t_lims=np.array([[-4.0, 4.0], [0.0, 1.0]], dtype="float32"),
            segment_color=segment_color,
            text_color=text_color,
            background_color=background_color,
            filled=filled,
            ticks_color=ticks_color,
        )
        self._attach(comp, parent_name)
        return comp

    def attach_playback_info(
        self,
        name: str | None = None,
        parent_name: str | None = None,
        fps: float | None = None,
        background_color: tuple[int, int, int] = (0, 0, 0),
        speed: float = 1.0,
        width: int = 300,
        height: int = 100,
    ) -> PlaybackInfoComponent:
        comp = PlaybackInfoComponent(
            name=name or self._auto_name(),
            fps=fps or self.default_fps,
            speed=speed,
            width=width,
            height=height,
            background_color=background_color,
        )
        self._attach(comp, parent_name)
        return comp

    def seek(self, t: float) -> None:
        """Jump to absolute time *t* seconds.

        Works both during playback and before `play()` (sets the start
        position).
        """
        if self._clock is not None:
            self._clock.seek(t)

    def seek_delta(self, delta_s: float) -> None:
        """Jump forward (positive) or backward (negative) by *delta_s*."""
        if self._clock is not None:
            self._clock.seek_delta(delta_s)

    def play(
        self,
        speed: float = 1.0,
        start_t: float = 0.0,
        seek_step: float = _SEEK_STEP_S,
    ) -> None:
        """Run the main playback loop.

        Args:
            speed: Global speed multiplier.
            start_t: Initial playback time in seconds.
            seek_step: Seconds to jump per arrow-key press (default 10).

        Controls:
            q       — quit
            Space   — pause / resume
            → arrow — seek forward  *seek_step* seconds
            ← arrow — seek backward *seek_step* seconds
        """
        self._sort_roots()
        self._clock = PlaybackClock(speed=speed)
        self._clock.seek(start_t)
        target_interval = 1.0 / self.default_fps

        try:
            while True:
                t = self._clock.t

                # Render every root component at the current time
                for root in self.roots:
                    frame = render_tree(root, t)
                    if frame is not None:
                        display_frame(root.name, frame)

                # --- Handle input ---
                # Wait just long enough to hit our target FPS
                wait_ms = max(1, int(target_interval * 1000))
                key = cv2.waitKeyEx(wait_ms)

                if key == _KEY_Q:
                    break
                elif key == _KEY_SPACE:
                    self._clock.toggle_pause()
                elif key in _KEYS_LEFT:
                    self._clock.seek_delta(-seek_step)
                elif key in _KEYS_RIGHT:
                    self._clock.seek_delta(seek_step)

        finally:
            self._cleanup()
            self._clock = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _sort_roots(self) -> None:
        order = {VideoComponent: 1, SkeletonComponent: 2}
        self.roots.sort(key=lambda c: order.get(type(c), 3))

    def _cleanup(self) -> None:
        cv2.destroyAllWindows()
        for root in self.roots:
            if isinstance(root, VideoComponent):
                root.close()

    @staticmethod
    def _auto_name() -> str:
        from uuid import uuid4
        return str(uuid4())


def _now_ms() -> float:
    return time() * 1000
