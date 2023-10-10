import numpy as np

from sign_language_tools.core.transform import Transform
import sign_language_tools.pose.transform.functional as F


class InterpolateMissing(Transform):
    """
    Compute missing landmarks (NaN coordinates) in a pose sequence using an interpolation function.
    See `get_landmark_interpolation_function` for more information about the interpolation. If the
    pose is only made of missing landmarks (nan), the pose is returned unchanged.

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
        jfink (v1.5 10.10.2023)

    """

    def __init__(self, method: str = "linear"):
        super().__init__()
        self.method = method

    def __call__(self, pose_sequence: np.ndarray):
        """
        Args:
            pose_sequence: Pose sequence containing observed and missing landmarks (NaN values).

        Returns:
            Pose sequence where missing landmarks have been replaced by interpolated landmarks.
        """
        if np.isnan(pose_sequence).all():
            return pose_sequence

        f = F.get_landmark_interpolation_function(pose_sequence, self.method)
        pose_sequence = f(np.arange(pose_sequence.shape[0]))
        g = F.get_landmark_interpolation_function(
            pose_sequence, method="nearest", extrapolate=True
        )
        missing_values_idxs = np.where(
            np.any(np.any(np.isnan(pose_sequence), axis=2), axis=1)
        )[0]
        pose_sequence[missing_values_idxs] = g(missing_values_idxs)
        return pose_sequence
