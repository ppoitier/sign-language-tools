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


class FixedResolutionNormalization(Transform):
    def __init__(self, width: int, height: int):
        super().__init__()
        self.resolution = np.array([width, height], dtype=np.float32)
        assert width > 0 and height > 0, "Both width and height must be greater than 0."

    def __call__(self, pose_sequence: np.ndarray) -> np.ndarray:
        """
        Normalizes a pose sequence to the range [-1, 1] based on a fixed resolution.

        @Args
            pose_sequence: A numpy array of shape (T, L, 2) where T is the number of frames,
                           L is the number of landmarks, and the last dimension holds (x, y).

        @Returns
            normalized_pose_sequence: The normalized pose sequence.
        """

        # Apply the normalization formula in a single vectorized operation
        # 1. Divide x by width and y by height
        # 2. Subtract 0.5 to center around zero
        # 3. Multiply by 2 to scale to the range [-1, 1]
        return 2 * ((pose_sequence / self.resolution) - 0.5)
