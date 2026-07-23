import numpy as np

from sign_language_tools.core.transform import Transform


class Clip(Transform):
    """Clips the coordinates of a pose sequence to a fixed value range.

    Any value below `min_value` is set to `min_value`, and any value above
    `max_value` is set to `max_value`.

    Args:
        min_value: Lower bound of the clipping range.
        max_value: Upper bound of the clipping range.

    Example:
        >>> import numpy as np
        >>> from sign_language_tools.pose.transform import Clip
        >>> pose_sequence = np.array([[[-0.2, 0.5, 1.3]]])  # (T, L, C)
        >>> transform = Clip(min_value=0.0, max_value=1.0)
        >>> transform(pose_sequence)
        array([[[0. , 0.5, 1. ]]])
    """

    def __init__(self, min_value: float = 0.0, max_value: float = 1.0):
        super().__init__()
        self.min = min_value
        self.max = max_value

    def __call__(self, pose_sequence: np.ndarray) -> np.ndarray:
        """Clips the pose sequence coordinates.

        Args:
            pose_sequence: Pose sequence of shape `(T, L, C)`, where `T` is
                the number of frames, `L` the number of landmarks, and `C`
                the number of coordinates per landmark.

        Returns:
            The clipped pose sequence, with the same shape as `pose_sequence`.
        """
        return np.clip(pose_sequence, self.min, self.max)
