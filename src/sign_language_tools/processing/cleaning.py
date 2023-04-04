import numpy as np


def remove_empty_landmark_frames(landmarks: np.ndarray):
    return landmarks[~np.isnan(landmarks).all(axis=1).any(axis=1)]
