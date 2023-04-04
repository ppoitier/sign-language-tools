import numpy as np

from sign_language_tools.processing.cleaning import remove_empty_landmark_frames
from sign_language_tools.features.landmarks.select import split_into_segments


def preprocess(landmarks: np.ndarray):
    landmarks = remove_empty_landmark_frames(landmarks)
    face, left_hand, pose, right_hand = split_into_segments(landmarks, ['face', 'hand', 'pose', 'hand'])

