from typing import Optional

import cv2
import mediapipe as mp
import numpy as np
from tqdm import tqdm

mp_holistic = mp.solutions.holistic


def extract_landmarks_from_video_file(
        video_path: str,
        options: Optional[dict] = None,
        show_progress: bool = False,
        frame_range: Optional[tuple[int, int]] = None,
):
    if options is None:
        options = {
            'static_image_mode': False,
            'model_complexity': 2,
            'refine_face_landmarks': True,
            'smooth_landmarks': True,
            'min_detection_confidence': 0.5,
            'min_tracking_confidence': 0.5,
            'enable_segmentation': False,
            'smooth_segmentation': False,
        }

    cap = cv2.VideoCapture(video_path)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_range is None:
        frame_range = range(n_frames)
    else:
        start, end = frame_range
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        frame_range = range(start, end)

    results = []
    with mp_holistic.Holistic(**options) as holistic:
        for frame_idx in tqdm(frame_range, disable=(not show_progress), unit='frame'):
            success, image = cap.read()
            if not success:
                tqdm.write(f'Could not read frame #{frame_idx}')
                break
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_results = holistic.process(image)
            results.append(image_results)

    cap.release()
    return results


def landmarks_to_np(landmarks, n_landmarks: int, dtype='float16', fill_value=np.nan):
    if landmarks is None:
        return np.full((n_landmarks, 3), fill_value=fill_value, dtype=dtype)
    return np.array([(lm.x, lm.y, lm.z) for lm in landmarks.landmark], dtype=dtype)
