import numpy as np

from sign_language_tools.core.transform import Transform


class GaussianNoise(Transform):
    """Adds Gaussian noise to every coordinate of a pose sequence.

    Args:
        scale: Standard deviation of the Gaussian noise added to each
            coordinate.

    Example:
        >>> import numpy as np
        >>> np.random.seed(0)
        >>> from sign_language_tools.pose.transform import GaussianNoise
        >>> pose_sequence = np.zeros((10, 5, 2))  # (T, L, C)
        >>> transform = GaussianNoise(scale=0.01)
        >>> transform(pose_sequence).shape
        (10, 5, 2)
    """

    def __init__(self, scale: float):
        super().__init__()
        self.scale = scale

    def __call__(self, pose_sequence: np.ndarray) -> np.ndarray:
        """Adds Gaussian noise to the pose sequence.

        Args:
            pose_sequence: Pose sequence of shape `(T, L, C)`, where `T` is
                the number of frames, `L` the number of landmarks, and `C`
                the number of coordinates per landmark.

        Returns:
            The pose sequence with added noise, with the same shape as
            `pose_sequence`.
        """
        noise = np.random.normal(scale=self.scale, size=pose_sequence.shape)
        return pose_sequence + noise
