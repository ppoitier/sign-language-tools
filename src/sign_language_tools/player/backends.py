"""Video decoding backends with random-access support.

The player needs two things from a video decoder:
  1. Read the frame at an arbitrary index — fast.
  2. Know the total frame count, fps, and dimensions.

Three backends are available (in order of preference):

  - **PyAV** (recommended): Python bindings to FFmpeg.  Seeks to the
    nearest keyframe, then decodes forward to the exact target frame.
    Actively maintained, no memory leaks, handles edge cases well.

  - **decord**: True random-access via a keyframe index.  Very fast for
    short clips, but has known memory leaks on long videos and occasional
    deadlocks.  Use only if you know your videos are short.

  - **OpenCV**: Always available.  Uses `CAP_PROP_POS_FRAMES` which can
    be slow for large backward jumps.  Good enough as a last resort.

Usage:
    backend = open_video("path/to/video.mp4")  # auto-selects best backend
    frame = backend.read_frame(42)              # BGR ndarray
    backend.close()
"""

from abc import ABC, abstractmethod

import numpy as np


class VideoBackend(ABC):
    """Uniform interface to a video file with random-access reads."""

    @property
    @abstractmethod
    def fps(self) -> float: ...

    @property
    @abstractmethod
    def width(self) -> int: ...

    @property
    @abstractmethod
    def height(self) -> int: ...

    @property
    @abstractmethod
    def frame_count(self) -> int: ...

    @property
    def duration_s(self) -> float:
        return self.frame_count / self.fps if self.fps > 0 else 0.0

    @abstractmethod
    def read_frame(self, index: int) -> np.ndarray | None:
        """Return the BGR frame at *index*, or None past the end."""
        ...

    @abstractmethod
    def close(self) -> None: ...

    def read_frame_at_time(self, t: float) -> np.ndarray | None:
        """Convenience: read the frame closest to time *t* seconds."""
        idx = max(0, min(self.frame_count - 1, round(t * self.fps)))
        return self.read_frame(idx)


# ======================================================================
# PyAV backend  (preferred — keyframe seek + precise decode)
# ======================================================================

class PyAVBackend(VideoBackend):
    """Uses PyAV (FFmpeg) for fast, memory-safe random access.

    Strategy: seek to the nearest keyframe *before* the target, then
    decode forward until we reach the desired frame.  This is the same
    approach that video editors use — it's fast for both small and large
    jumps and doesn't leak memory.

    For sequential reads (frame N then N+1) we skip the seek entirely
    and just decode the next frame, which is essentially free.
    """

    def __init__(self, filepath: str):
        import av  # type: ignore[import-untyped]

        self._container = av.open(filepath)
        self._stream = self._container.streams.video[0]

        # Enable multi-threaded decoding for speed
        self._stream.thread_type = "AUTO"

        self._fps_val = float(self._stream.average_rate or self._stream.rate or 25)
        self._width_val = self._stream.codec_context.width
        self._height_val = self._stream.codec_context.height
        self._frame_count_val = self._stream.frames or self._estimate_frame_count()
        self._time_base = float(self._stream.time_base)

        # Cache: avoid re-seeking when reading sequentially
        self._last_decoded_idx: int = -1
        self._last_frame: np.ndarray | None = None

    def _estimate_frame_count(self) -> int:
        """Fallback when the container doesn't report frame count."""
        if self._stream.duration and self._stream.time_base:
            duration_s = float(self._stream.duration * self._stream.time_base)
            return round(duration_s * self._fps_val)
        return 0

    @property
    def fps(self) -> float:
        return self._fps_val

    @property
    def width(self) -> int:
        return self._width_val

    @property
    def height(self) -> int:
        return self._height_val

    @property
    def frame_count(self) -> int:
        return self._frame_count_val

    def read_frame(self, index: int) -> np.ndarray | None:
        if index < 0:
            return None

        # Fast path: if we just decoded this frame, return the cached copy
        if index == self._last_decoded_idx and self._last_frame is not None:
            return self._last_frame

        # Fast path: sequential read — just grab the next decoded frame
        if index == self._last_decoded_idx + 1:
            frame = self._decode_next()
            if frame is not None:
                self._last_decoded_idx = index
                self._last_frame = frame
            return frame

        # Seek to nearest keyframe before target, then decode forward
        target_pts = int(index / self._fps_val / self._time_base)
        self._container.seek(target_pts, stream=self._stream)

        frame = None
        for packet in self._container.demux(self._stream):
            for video_frame in packet.decode():
                frame_idx = round(float(video_frame.pts * self._time_base) * self._fps_val)
                bgr = video_frame.to_ndarray(format="bgr24")
                if frame_idx >= index:
                    self._last_decoded_idx = frame_idx
                    self._last_frame = bgr
                    return bgr
                # Keep the last decoded frame in case we overshoot
                frame = bgr

        # If we fell through (end of stream), return whatever we last decoded
        if frame is not None:
            self._last_decoded_idx = index
            self._last_frame = frame
        return frame

    def _decode_next(self) -> np.ndarray | None:
        """Decode exactly one frame (no seek)."""
        for packet in self._container.demux(self._stream):
            for video_frame in packet.decode():
                return video_frame.to_ndarray(format="bgr24")
        return None

    def close(self) -> None:
        self._container.close()
        self._last_frame = None


# ======================================================================
# decord backend  (fast but leaks memory on long videos)
# ======================================================================

class DecordBackend(VideoBackend):
    """Uses `decord.VideoReader` for efficient random-access decoding.

    decord builds a keyframe index on open, so `reader[idx]` is fast.

    **Warning**: decord has known memory leaks on long videos (especially
    on Linux) and occasional deadlocks with multi-threading.  Prefer
    PyAVBackend for long recordings.
    """

    def __init__(self, filepath: str):
        from decord import VideoReader, cpu  # type: ignore[import-untyped]

        self._reader = VideoReader(filepath, ctx=cpu(0), num_threads=1)
        self._fps_val = float(self._reader.get_avg_fps())

    @property
    def fps(self) -> float:
        return self._fps_val

    @property
    def width(self) -> int:
        return self._reader[0].shape[1]

    @property
    def height(self) -> int:
        return self._reader[0].shape[0]

    @property
    def frame_count(self) -> int:
        return len(self._reader)

    def read_frame(self, index: int) -> np.ndarray | None:
        if index < 0 or index >= len(self._reader):
            return None
        # decord returns RGB — convert to BGR for OpenCV display
        frame_rgb = self._reader[index].asnumpy()
        return frame_rgb[:, :, ::-1].copy()

    def close(self) -> None:
        del self._reader


# ======================================================================
# OpenCV backend  (fallback — works everywhere, slower seeks)
# ======================================================================

class OpenCVBackend(VideoBackend):
    """Falls back to `cv2.VideoCapture` when nothing else is available.

    Seeking uses `CAP_PROP_POS_FRAMES` which may re-decode from the
    nearest keyframe.  For short seeks or sequential reads this is fine;
    for large jumps on long videos it can be noticeably slower.
    """

    def __init__(self, filepath: str):
        import cv2

        self._cap = cv2.VideoCapture(filepath)
        if not self._cap.isOpened():
            raise IOError(f"OpenCV could not open: {filepath}")
        self._fps_val = self._cap.get(cv2.CAP_PROP_FPS)
        self._width_val = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._height_val = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._frame_count_val = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._last_read_idx: int = -1

    @property
    def fps(self) -> float:
        return self._fps_val

    @property
    def width(self) -> int:
        return self._width_val

    @property
    def height(self) -> int:
        return self._height_val

    @property
    def frame_count(self) -> int:
        return self._frame_count_val

    def read_frame(self, index: int) -> np.ndarray | None:
        if index < 0 or index >= self._frame_count_val:
            return None
        # Only seek if we're not already at the next sequential frame
        if index != self._last_read_idx + 1:
            import cv2
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = self._cap.read()
        if not ok or frame is None:
            return None
        self._last_read_idx = index
        return frame

    def close(self) -> None:
        self._cap.release()


class TensorBackend(VideoBackend):
    """Wraps an in-memory array of frames with shape ``(T, H, W, C)``.

    Accepts either a **numpy ndarray** or a **PyTorch tensor** (or
    anything that supports indexing and has a ``.shape`` attribute).
    PyTorch is only imported if the input is not already a numpy array.

    The frames are expected in **RGB** or **BGR** channel order; use
    *is_rgb* to indicate which (default ``True`` for PyTorch convention).
    Conversion to BGR (for OpenCV display) happens per-frame on read.

    This backend is ideal for:
      - Visualizing model outputs (generated / reconstructed video).
      - Displaying preloaded tensors without writing to disk first.
      - Quick debugging of tensor-shaped data.

    Example::

        # From a PyTorch tensor (T, H, W, 3)
        backend = TensorBackend(my_tensor, fps=25)

        # From a numpy array already in BGR
        backend = TensorBackend(frames_bgr, fps=30, is_rgb=False)
    """

    def __init__(
        self,
        frames: np.ndarray,
        fps: float = 25.0,
        is_rgb: bool = True,
    ):
        if not isinstance(frames, np.ndarray):
            if hasattr(frames, "detach") and hasattr(frames, "cpu"):
                frames = frames.detach().cpu().numpy()
            else:
                frames = np.asarray(frames)

        if frames.ndim != 4 or frames.shape[-1] not in (1, 3):
            raise ValueError(
                f"Expected frames with shape (T, H, W, C) where C in {{1, 3}}, "
                f"got {frames.shape}"
            )

        self._frames = frames.astype("uint8") if frames.dtype != np.uint8 else frames
        self._is_rgb = is_rgb
        self._fps_val = fps

    @property
    def fps(self) -> float:
        return self._fps_val

    @property
    def width(self) -> int:
        return self._frames.shape[2]

    @property
    def height(self) -> int:
        return self._frames.shape[1]

    @property
    def frame_count(self) -> int:
        return self._frames.shape[0]

    def read_frame(self, index: int) -> np.ndarray | None:
        if index < 0 or index >= self._frames.shape[0]:
            return None
        frame = self._frames[index]
        if self._is_rgb and frame.shape[-1] == 3:
            frame = frame[:, :, ::-1].copy()  # RGB → BGR
        return frame

    def close(self) -> None:
        # Nothing to release — the caller owns the tensor
        self._frames = np.empty((0, 0, 0, 0), dtype="uint8")


# ======================================================================
# Factory
# ======================================================================

_BACKEND_ORDER = {
    "auto": ["pyav", "decord", "opencv"],
    "pyav": ["pyav"],
    "decord": ["decord"],
    "opencv": ["opencv"],
}

_BACKEND_CLASSES = {
    "pyav": PyAVBackend,
    "decord": DecordBackend,
    "opencv": OpenCVBackend,
}


def open_video(filepath: str, backend: str = "auto") -> VideoBackend:
    """Open a video file with the best available backend.

    Args:
        filepath: Path to the video file.
        backend: ``"pyav"``, ``"decord"``, ``"opencv"``, or ``"auto"``
            (tries pyav → decord → opencv).
    """
    candidates = _BACKEND_ORDER.get(backend)
    if candidates is None:
        raise ValueError(
            f"Unknown backend: '{backend}'. Choose from: {list(_BACKEND_ORDER)}"
        )

    last_error: Exception | None = None
    for name in candidates:
        try:
            return _BACKEND_CLASSES[name](filepath)
        except ImportError as e:
            last_error = e
            continue

    raise ImportError(
        f"No video backend available (tried {candidates}). "
        f"Install one of: pip install av, pip install decord, or pip install opencv-python. "
        f"Last error: {last_error}"
    )
