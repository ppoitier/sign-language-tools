import numpy as np

from sign_language_tools.core.transform import Transform


class Scale(Transform):
    """Scales a pose sequence around a fixed center point.

    Multiplies every coordinate by `scaling_factor`, then re-centers the
    result so that `center` stays fixed.

    Args:
        scaling_factor: Factor applied to every coordinate.
        center: `(x, y)` reference point that stays fixed by the scaling.

    Example:
        >>> import numpy as np
        >>> from sign_language_tools.pose.transform import Scale
        >>> pose_sequence = np.array([[[0.5, 0.5], [1.0, 1.0]]])  # (T, L, C)
        >>> transform = Scale(scaling_factor=2.0)
        >>> result = transform(pose_sequence)
        >>> result[0, 0]  # the center point is left unchanged
        array([0.5, 0.5])
        >>> result[0, 1]  # points further from the center move further away
        array([1.5, 1.5])
    """

    def __init__(self, scaling_factor: float, center=(0.5, 0.5)):
        super().__init__()
        self.center = center
        self.dx = (scaling_factor * center[0]) - center[0]
        self.dy = (scaling_factor * center[1]) - center[1]
        self.scaling_factor = scaling_factor

    def __call__(self, pose_sequence: np.ndarray) -> np.ndarray:
        """Scales the pose sequence.

        Args:
            pose_sequence: Pose sequence of shape `(T, L, C)`, where `T` is
                the number of frames, `L` the number of landmarks, and `C`
                the number of coordinates per landmark.

        Returns:
            The scaled pose sequence.
        """
        pose_sequence = pose_sequence.copy()
        pose_sequence[:, :, :] *= self.scaling_factor
        pose_sequence[:, :, 0] -= self.dx
        pose_sequence[:, :, 1] -= self.dy
        return pose_sequence


class RandomScale(Scale):
    """Scales a pose sequence by a randomly chosen factor around a fixed center.

    Each time the transform is called, the scaling factor is drawn
    uniformly at random between `min_scale` and `max_scale`.

    Args:
        min_scale: Lower bound for the random scaling factor.
        max_scale: Upper bound for the random scaling factor.
        center: `(x, y)` reference point that stays fixed by the scaling.

    Example:
        >>> import numpy as np
        >>> from sign_language_tools.pose.transform import RandomScale
        >>> pose_sequence = np.random.rand(10, 5, 2)  # (T, L, C)
        >>> transform = RandomScale(min_scale=0.8, max_scale=1.2)
        >>> transform(pose_sequence).shape
        (10, 5, 2)
    """

    def __init__(self, min_scale: float, max_scale: float, center=(0.5, 0.5)):
        super().__init__(1.0, center)
        self.min_scale = min_scale
        self.max_scale = max_scale

    def __call__(self, pose_sequence: np.ndarray) -> np.ndarray:
        """Scales the pose sequence by a randomly chosen factor.

        Args:
            pose_sequence: Pose sequence of shape `(T, L, C)`, where `T` is
                the number of frames, `L` the number of landmarks, and `C`
                the number of coordinates per landmark.

        Returns:
            The scaled pose sequence.
        """
        scaling_factor = np.random.uniform(self.min_scale, self.max_scale)
        self.dx = (scaling_factor * self.center[0]) - self.center[0]
        self.dy = (scaling_factor * self.center[1]) - self.center[1]
        self.scaling_factor = scaling_factor
        return super().__call__(pose_sequence)

