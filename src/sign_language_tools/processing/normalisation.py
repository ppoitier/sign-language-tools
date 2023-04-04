import numpy as np


def std_scale_landmarks(landmarks: np.ndarray):
    mean = np.nanmean(landmarks)
    std = np.nanstd(landmarks)
    return (landmarks - mean) / std
