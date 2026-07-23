import numpy as np
import random

from sign_language_tools.core.transform import Transform


class Translation(Transform):
    """Translates a pose sequence by a fixed offset.

    Adds `dx` to the x-coordinate and `dy` to the y-coordinate of every
    landmark.

    Args:
        dx: Offset added to the x-coordinate.
        dy: Offset added to the y-coordinate.

    Example:
        >>> import numpy as np
        >>> from sign_language_tools.pose.transform import Translation
        >>> pose_sequence = np.zeros((1, 1, 2))  # (T, L, C)
        >>> transform = Translation(dx=0.1, dy=-0.2)
        >>> result = transform(pose_sequence)
        >>> tuple(result[0, 0])
        (0.1, -0.2)
    """

    def __init__(self, dx: float, dy: float):
        super().__init__()
        self.dx = dx
        self.dy = dy

    def __call__(self, pose_sequence: np.ndarray) -> np.ndarray:
        """Translates the pose sequence.

        Args:
            pose_sequence: Pose sequence of shape `(T, L, C)`, where `T` is
                the number of frames, `L` the number of landmarks, and `C`
                the number of coordinates per landmark.

        Returns:
            The translated pose sequence.
        """
        pose_sequence = pose_sequence.copy()
        pose_sequence[:, :, 0] += self.dx
        pose_sequence[:, :, 1] += self.dy
        return pose_sequence


class RandomTranslation(Transform):
    """Translates a pose sequence by a randomly chosen offset.

    Each time the transform is called, the x and y offsets are drawn
    uniformly at random from `dx_range` and `dy_range` respectively.

    Args:
        dx_range: `(min, max)` range for the random x offset.
        dy_range: `(min, max)` range for the random y offset.

    Example:
        >>> import numpy as np
        >>> from sign_language_tools.pose.transform import RandomTranslation
        >>> pose_sequence = np.zeros((10, 5, 2))  # (T, L, C)
        >>> transform = RandomTranslation(dx_range=(-0.2, 0.2), dy_range=(-0.2, 0.2))
        >>> transform(pose_sequence).shape
        (10, 5, 2)
    """

    def __init__(self, dx_range=(-0.2, 0.2), dy_range=(-0.2, 0.2)):
        super().__init__()
        self.dx_range = dx_range
        self.dy_range = dy_range

    def __call__(self, pose_sequence: np.ndarray) -> np.ndarray:
        """Translates the pose sequence by a randomly chosen offset.

        Args:
            pose_sequence: Pose sequence of shape `(T, L, C)`, where `T` is
                the number of frames, `L` the number of landmarks, and `C`
                the number of coordinates per landmark.

        Returns:
            The translated pose sequence.
        """
        pose_sequence = pose_sequence.copy()
        pose_sequence[:, :, 0] += random.uniform(*self.dx_range)
        pose_sequence[:, :, 1] += random.uniform(*self.dy_range)
        return pose_sequence
