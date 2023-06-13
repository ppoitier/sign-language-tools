import numpy as np


class FilterEmpty:
    def __call__(self, pose_sequence: np.ndarray) -> np.ndarray:
        return pose_sequence[~np.isnan(pose_sequence).all(axis=1).any(axis=1)]
