import numpy as np

from sign_language_tools.core.transform import Transform


class FilterEmpty(Transform):
    def __call__(self, pose_sequence: np.ndarray) -> np.ndarray:
        return pose_sequence[~np.isnan(pose_sequence).all(axis=1).any(axis=1)]


class FilterLandmarks(Transform):
    def __init__(self, filtered_indices: set[int] | list[int]):
        super().__init__()
        self.filtered_indices = set(filtered_indices)

    def __call__(self, pose_sequence: np.ndarray) -> np.ndarray:
        T, L, D = pose_sequence.shape
        lm_indices = [idx for idx in range(L) if idx not in self.filtered_indices]
        return pose_sequence[:, lm_indices]
