import numpy as np

from sign_language_tools.core.transform import Transform
import sign_language_tools.pose.transforms.functional as F


class InterpolateMissing(Transform):
    """Fills missing landmarks (NaN coordinates) in a pose sequence by interpolation.

    Builds an interpolation function over the frame axis (see
    `get_landmark_interpolation_function`) to fill in NaN landmarks from
    observed ones, then fills any values still missing after that (e.g.
    leading/trailing frames) using nearest-neighbor extrapolation. If the
    whole pose sequence is NaN, it is returned unchanged.

    Args:
        method: Interpolation method used for observed landmarks. One of
            `"linear"`, `"nearest"`, `"previous"`, `"next"`.

    Example:
        >>> import numpy as np
        >>> from sign_language_tools.pose.transform import InterpolateMissing
        >>> pose_sequence = np.array([[[0.0]], [[np.nan]], [[2.0]]])  # (T, L, C)
        >>> transform = InterpolateMissing(method="linear")
        >>> transform(pose_sequence).flatten()
        array([0., 1., 2.])
    """

    def __init__(self, method: str = "linear"):
        super().__init__()
        self.method = method

    def __call__(self, pose_sequence: np.ndarray) -> np.ndarray:
        """Fills missing landmarks in the pose sequence.

        Args:
            pose_sequence: Pose sequence of shape `(T, L, C)` containing
                observed and missing (NaN) landmarks.

        Returns:
            The pose sequence with missing landmarks replaced by
            interpolated values, with the same shape as `pose_sequence`
            (or unchanged if it is entirely missing).
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
