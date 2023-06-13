import numpy as np

import sign_language_tools.pose.transform.functional as F


class InterpolateMissing:
    """
    Compute missing landmarks (NaN coordinates) in a pose sequence using an interpolation function.
    See `get_landmark_interpolation_function` for more information about the interpolation.

    Args:
        method: Specify the method used to compute the interpolated landmarks.
            - `linear`
            - `nearest`
            - `previous`
            - `next`

    Returns:
        interpolated_landmarks: Tensor containing observed and interpolated landmarks.

    Author:
        ppoitier (v1 03.04.2023)
    """
    def __init__(self, method: str = 'linear'):
        self.method = method

    def __call__(self, pose_sequence: np.ndarray):
        """
        Args:
            pose_sequence: Pose sequence containing observed and missing landmarks (NaN values).

        Returns:
            Pose sequence where missing landmarks have been replaced by interpolated landmarks.
        """
        f = F.get_landmark_interpolation_function(pose_sequence, self.method)
        return f(np.arange(pose_sequence.shape[0]))
