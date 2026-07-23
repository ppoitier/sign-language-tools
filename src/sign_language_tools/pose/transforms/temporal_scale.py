import math
import random

import numpy as np

from sign_language_tools.pose.transforms.resample import Resample


class TemporalScale(Resample):
    """Resamples a pose sequence to a length scaled by a fixed factor.

    The target length is computed as `ceil(T * scale)`, where `T` is the
    size of the pose sequence along `time_axis`. The actual resampling is
    delegated to `Resample`.

    Args:
        scale: Factor applied to the sequence length. Values above `1.0`
            stretch the sequence (more frames), values below `1.0`
            compress it (fewer frames).
        time_axis: Axis of the pose sequence along which the length is
            measured and scaled.

    Example:
        >>> import numpy as np
        >>> from sign_language_tools.pose.transform import TemporalScale
        >>> pose_sequence = np.random.rand(10, 5, 2)  # (T, L, C)
        >>> transform = TemporalScale(scale=1.5)
        >>> transform(pose_sequence).shape
        (15, 5, 2)
    """

    def __init__(self, scale: float, time_axis=0):
        super().__init__(new_length=0)
        self.scale = scale
        self.time_axis = time_axis

    def __call__(self, pose_sequence: np.ndarray) -> np.ndarray:
        """Resamples the pose sequence to its scaled length.

        Args:
            pose_sequence: Pose sequence of shape `(T, L, C)`, where `T` is
                the number of frames, `L` the number of landmarks, and `C`
                the number of coordinates per landmark.

        Returns:
            The resampled pose sequence, of shape `(ceil(T * scale), L, C)`.
        """
        self.new_length = math.ceil(pose_sequence.shape[self.time_axis] * self.scale)
        return super().__call__(pose_sequence)


class RandomTemporalScale(Resample):
    """Resamples a pose sequence to a length scaled by a randomly chosen factor.

    Each time the transform is called, the scaling factor is drawn
    uniformly at random between `min_scale` and `max_scale`, and the target
    length is computed as `ceil(T * scale)`.

    Args:
        min_scale: Lower bound for the random scaling factor.
        max_scale: Upper bound for the random scaling factor.
        time_axis: Axis of the pose sequence along which the length is
            measured and scaled.

    Example:
        >>> import numpy as np
        >>> from sign_language_tools.pose.transform import RandomTemporalScale
        >>> pose_sequence = np.random.rand(10, 5, 2)  # (T, L, C)
        >>> transform = RandomTemporalScale(min_scale=0.5, max_scale=1.5)
        >>> transform(pose_sequence).shape[1:]
        (5, 2)
    """

    def __init__(self, min_scale: float, max_scale: float, time_axis=0):
        super().__init__(new_length=0)
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.time_axis = time_axis

    def __call__(self, pose_sequence: np.ndarray) -> np.ndarray:
        """Resamples the pose sequence to a randomly scaled length.

        Args:
            pose_sequence: Pose sequence of shape `(T, L, C)`, where `T` is
                the number of frames, `L` the number of landmarks, and `C`
                the number of coordinates per landmark.

        Returns:
            The resampled pose sequence, of shape `(ceil(T * scale), L, C)`
            where `scale` is randomly drawn between `min_scale` and
            `max_scale`.
        """
        scale = random.uniform(self.min_scale, self.max_scale)
        self.new_length = math.ceil(pose_sequence.shape[self.time_axis] * scale)
        return super().__call__(pose_sequence)

