import numpy as np

from sign_language_tools.core.transform import Transform


class Standardization(Transform):
    def __call__(self, pose_sequence: np.ndarray) -> np.ndarray:
        mean = np.nanmean(pose_sequence, axis=(0, 1))
        std = np.nanstd(pose_sequence, axis=(0, 1))
        return np.divide(
            pose_sequence - mean, std, where=std != 0, out=np.zeros_like(pose_sequence)
        )


class MinMaxNormalization(Transform):
    def __call__(self, pose_sequence: np.ndarray) -> np.ndarray:
        min_val = np.nanmin(pose_sequence, axis=(0, 1))
        max_val = np.nanmax(pose_sequence, axis=(0, 1))
        scaling_factors = (max_val - min_val).reshape(1, 1, -1)
        return np.divide(
            pose_sequence - min_val,
            scaling_factors,
            where=scaling_factors != 0,
            out=np.zeros_like(pose_sequence),
        )
