import numpy as np


class Standardization:
    def __call__(self, pose_sequence: np.ndarray) -> np.ndarray:
        mean = np.nanmean(pose_sequence)
        std = np.nanstd(pose_sequence)
        return (pose_sequence - mean) / std


class MinMaxNormalization:
    def __call__(self, pose_sequence: np.ndarray) -> np.ndarray:
        min_val = np.nanmin(pose_sequence)
        max_val = np.nanmax(pose_sequence)
        return (pose_sequence - min_val) / (max_val - min_val)
