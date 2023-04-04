import numpy as np

from sign_language_tools.processing.cleaning import remove_empty_landmark_frames
from sign_language_tools.features.landmarks.select import split_into_segments
from sign_language_tools.features.landmarks.indices import FACE_LIPS_INDICES
from sign_language_tools.processing.resampling import resample
from sign_language_tools.processing.padding import pad
from sign_language_tools.processing.interpolation import interpolate_missing_landmarks, fill_empty_landmark_sequences


def keep_only_frames_of_interest(landmarks):
    face, left_hand, pose, right_hand = split_into_segments(landmarks, ['face', 'hand', 'pose', 'hand'])
    lips = face[:, FACE_LIPS_INDICES]
    return np.concatenate([lips, left_hand, pose, right_hand], axis=1)


def preprocess(landmarks: np.ndarray, max_length: int):
    landmarks = keep_only_frames_of_interest(landmarks)
    landmarks = remove_empty_landmark_frames(landmarks)
    landmarks = fill_empty_landmark_sequences(landmarks, 0.0)

    segments = split_into_segments(landmarks, ['lips', 'hand', 'pose', 'hand'])
    segments = [interpolate_missing_landmarks(s, method='linear') for s in segments]

    landmarks = np.concatenate(segments, axis=1)
    t, n, d = landmarks.shape

    if t < max_length:
        landmarks = pad(landmarks, max_length, mode='edge')
    elif t > max_length:
        landmarks = resample(landmarks, max_length, method='linear')

    return landmarks
