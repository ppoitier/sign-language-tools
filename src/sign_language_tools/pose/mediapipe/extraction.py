from collections import defaultdict

import numpy as np
import mediapipe as mp
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision import (
    HolisticLandmarkerOptions,
    HolisticLandmarker,
    HolisticLandmarkerResult,
    RunningMode,
)

from sign_language_tools.video.decoding import iterate_video_frames_using_vidgear


def load_holistic_landmarker(model_path: str, use_gpu: bool = False, options: HolisticLandmarkerOptions | None = None):
    """Load a MediaPipe holistic landmarker for video inference.

    Args:
        model_path (str): Path to the holistic landmarker `.task` model file.
        use_gpu (bool): Whether to run inference on GPU instead of CPU. Defaults to `False`.
        options (HolisticLandmarkerOptions | None): Custom landmarker options. If `None`,
            default options are used with the model running in `VIDEO` mode. Defaults to `None`.

    Returns:
        HolisticLandmarker: The initialized holistic landmarker, ready to process video frames.
    """
    base_options = BaseOptions(
        model_asset_path=model_path,
        delegate=BaseOptions.Delegate.GPU if use_gpu else BaseOptions.Delegate.CPU,
    )
    options = HolisticLandmarkerOptions(
        base_options=base_options,
        running_mode=RunningMode.VIDEO,
        min_face_detection_confidence=0.5,
        min_face_suppression_threshold=0.5,
        min_face_landmarks_confidence=0.5,
        min_pose_detection_confidence=0.2,
        min_pose_suppression_threshold=0.5,
        min_pose_landmarks_confidence=0.2,
        min_hand_landmarks_confidence=0.2,
    ) if options is None else options
    return HolisticLandmarker.create_from_options(options)


def _landmarks_to_array(landmarks, n_expected_landmarks: int) -> np.ndarray:
    """Convert a list of MediaPipe landmarks into an `(L, 3)` array.

    Args:
        landmarks: Sequence of MediaPipe landmark objects, each exposing `x`, `y` and `z`.
        n_expected_landmarks (int): Number of landmarks expected for this landmark group
            (e.g. 33 for pose, 21 for a hand, 478 for the face). If `landmarks` does not
            contain exactly this many entries (e.g. because detection failed for the frame),
            an array filled with `NaN` is returned instead.

    Returns:
        np.ndarray: Array of shape `(L, C)` with `L=n_expected_landmarks` and `C=3` (x, y, z),
            dtype `float16`.
    """
    if len(landmarks) != n_expected_landmarks:
        return np.full((n_expected_landmarks, 3), np.nan, dtype="float16")
    array = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype="float16")
    return array


def extract_poses_from_video_file(
    video_path: str,
    holistic_landmarker: HolisticLandmarker,
    show_progress=False,
) -> dict[str, np.ndarray]:
    """Extract holistic pose landmarks from every frame of a video file.

    Args:
        video_path (str): Path to the video file to process.
        holistic_landmarker (HolisticLandmarker): Landmarker used to run detection on each
            frame, e.g. as returned by [`load_holistic_landmarker`][sign_language_tools.pose.mediapipe.extraction.load_holistic_landmarker].
        show_progress (bool): Whether to display a progress bar while iterating over the
            video frames. Defaults to `False`.

    Returns:
        dict[str, np.ndarray]: Mapping from landmark group (`"pose"`, `"left_hand"`,
            `"right_hand"`, `"face"`) to an array of shape `(T, L, C)`, with `T` the number
            of frames, `L` the number of landmarks in the group and `C=3` (x, y, z).
    """
    poses = defaultdict(list)
    for idx, (timestamp_ms, frame) in enumerate(
        iterate_video_frames_using_vidgear(video_path, show_progress=show_progress)
    ):
        mp_img = mp.Image(mp.ImageFormat.SRGB, frame)
        results: HolisticLandmarkerResult = holistic_landmarker.detect_for_video(
            mp_img, timestamp_ms
        )

        poses["pose"].append(
            _landmarks_to_array(results.pose_landmarks, n_expected_landmarks=33)
        )
        poses["left_hand"].append(
            _landmarks_to_array(results.left_hand_landmarks, n_expected_landmarks=21)
        )
        poses["right_hand"].append(
            _landmarks_to_array(results.right_hand_landmarks, n_expected_landmarks=21)
        )
        poses["face"].append(
            _landmarks_to_array(results.face_landmarks, n_expected_landmarks=478)
        )

    return {k: np.stack(v, axis=0) for k, v in poses.items()}