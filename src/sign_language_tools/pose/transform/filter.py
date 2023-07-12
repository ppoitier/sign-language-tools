import numpy as np

from sign_language_tools.core.transform import Transform


class FilterEmpty(Transform):
    def __call__(self, pose_sequence: np.ndarray) -> np.ndarray:
        return pose_sequence[~np.isnan(pose_sequence).all(axis=1).any(axis=1)]
