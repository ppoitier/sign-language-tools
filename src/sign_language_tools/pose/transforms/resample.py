import numpy as np

import sign_language_tools.pose.transforms.functional as F
from sign_language_tools.core.transform import Transform


class Resample(Transform):
    """Resamples a pose sequence to a fixed number of frames.

    Uses interpolation over the frame axis to compute new, evenly spaced
    frames. See `get_landmark_interpolation_function`.

    Args:
        new_length: Number of frames in the resampled pose sequence.
        method: Interpolation method used to compute new frames. One of
            `"linear"`, `"nearest"`, `"previous"`, `"next"`.

    Example:
        >>> import numpy as np
        >>> from sign_language_tools.pose.transform import Resample
        >>> pose_sequence = np.random.rand(10, 5, 2)  # (T, L, C)
        >>> transform = Resample(new_length=20)
        >>> transform(pose_sequence).shape
        (20, 5, 2)
    """

    def __init__(self, new_length: int, method: str = 'linear'):
        super().__init__()
        self.new_length = new_length
        self.method = method

    def __call__(self, pose_sequence: np.ndarray) -> np.ndarray:
        """Resamples the pose sequence.

        Args:
            pose_sequence: Pose sequence of shape `(T, L, C)`, where `T` is
                the number of frames, `L` the number of landmarks, and `C`
                the number of coordinates per landmark.

        Returns:
            The resampled pose sequence, of shape `(new_length, L, C)`.
        """
        t = pose_sequence.shape[0]
        x = np.linspace(0, t - 1, self.new_length)
        f = F.get_landmark_interpolation_function(pose_sequence, self.method)
        return f(x)


class RandomResample(Resample):
    """Resamples a pose sequence to a randomly chosen number of frames.

    Each time the transform is called, the target length is drawn
    uniformly at random between `min_length` (inclusive) and `max_length`
    (exclusive).

    Args:
        min_length: Lower bound (inclusive) for the random target length.
        max_length: Upper bound (exclusive) for the random target length.
        method: Interpolation method used to compute new frames. One of
            `"linear"`, `"nearest"`, `"previous"`, `"next"`.

    Example:
        >>> import numpy as np
        >>> from sign_language_tools.pose.transform import RandomResample
        >>> pose_sequence = np.random.rand(10, 5, 2)  # (T, L, C)
        >>> transform = RandomResample(min_length=5, max_length=15)
        >>> new_pose_sequence = transform(pose_sequence)
        >>> new_pose_sequence.shape[1:]
        (5, 2)
        >>> 5 <= new_pose_sequence.shape[0] < 15
        True
    """

    def __init__(self, min_length: int, max_length: int, method: str = 'linear'):
        super().__init__(min_length, method)
        self.min_length = min_length
        self.max_length = max_length

    def __call__(self, pose_sequence: np.ndarray) -> np.ndarray:
        """Resamples the pose sequence to a randomly chosen length.

        Args:
            pose_sequence: Pose sequence of shape `(T, L, C)`, where `T` is
                the number of frames, `L` the number of landmarks, and `C`
                the number of coordinates per landmark.

        Returns:
            The resampled pose sequence, of shape `(new_length, L, C)`,
            where `new_length` is randomly drawn between `min_length` and
            `max_length`.
        """
        self.new_length = np.random.randint(self.min_length, self.max_length)
        return super().__call__(pose_sequence)
