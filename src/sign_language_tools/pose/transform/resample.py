import numpy as np

import sign_language_tools.pose.transform.functional as F
from sign_language_tools.core.transform import Transform


class Resample(Transform):
    """Resample a sequence of landmarks to a given length.

    Interpolation is used to compute new landmarks. See `interpolation.get_landmark_interpolation_function`.

    Args:
        new_length: The new length to resample.
        method: The method used to interpolate landmarks.

    Returns:
        resampled_landmarks: The sequence of landmarks that has been resampled to `new_length`.
    """

    def __init__(self, new_length: int, method: str = 'linear'):
        super().__init__()
        self.new_length = new_length
        self.method = method

    def __call__(self, pose_sequence: np.ndarray) -> np.ndarray:
        # landmarks shape (T, N, C)
        t = pose_sequence.shape[0]
        x = np.linspace(0, t - 1, self.new_length)
        f = F.get_landmark_interpolation_function(pose_sequence, self.method)
        return f(x)
