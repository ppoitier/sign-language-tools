from typing import Iterator
import numpy as np
from tqdm import tqdm
import cv2
from vidgear.gears import CamGear


def iterate_video_frames_using_vidgear(video_path: str, show_progress: bool = False) -> Iterator[tuple[int, np.ndarray]]:
    """Iterates over the frames of a video file, decoded with `vidgear`'s `CamGear`.

    Args:
        video_path (str): Path to the video file to decode.
        show_progress (bool): Whether to display a `tqdm` progress bar while iterating.

    Yields:
        tuple[int, np.ndarray]: A `(timestamp_ms, frame)` pair for each decoded frame,
            where `timestamp_ms` is the frame's estimated timestamp in milliseconds
            (computed from the frame index and the video's FPS) and `frame` is an
            RGB image array with shape `(H, W, 3)`.

    Raises:
        ZeroDivisionError: If the video's reported FPS is `0` (can happen with some
            codecs/backends), since timestamps are computed as `frame_nb * 1000 / fps`.
    """
    capture = CamGear(source=video_path).start()
    n_frames = int(capture.stream.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.stream.get(cv2.CAP_PROP_FPS))
    progress_bar = tqdm(range(n_frames), disable=not show_progress)
    for frame_nb in progress_bar:
        timestamp_ms = int(round(frame_nb * 1000 / fps))
        frame = capture.read()
        if frame is None:
            progress_bar.write(f"Cannot read frame {frame_nb}.")
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        yield timestamp_ms, frame
    capture.stop()
