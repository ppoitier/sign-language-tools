import numpy as np

from sign_language_tools.core.transform import Transform


class Standardization(Transform):
    """Standardizes a pose sequence to zero mean and unit variance per coordinate.

    For each coordinate channel (e.g. x, y, z), computes the mean and
    standard deviation across all frames and landmarks (ignoring NaNs), and
    rescales the pose sequence accordingly. Channels with zero variance are
    left at zero instead of dividing by zero.

    Example:
        >>> import numpy as np
        >>> from sign_language_tools.pose.transform import Standardization
        >>> pose_sequence = np.random.rand(10, 5, 2)  # (T, L, C)
        >>> transform = Standardization()
        >>> transform(pose_sequence).shape
        (10, 5, 2)
    """

    def __call__(self, pose_sequence: np.ndarray) -> np.ndarray:
        """Standardizes the pose sequence.

        Args:
            pose_sequence: Pose sequence of shape `(T, L, C)`, where `T` is
                the number of frames, `L` the number of landmarks, and `C`
                the number of coordinates per landmark.

        Returns:
            The standardized pose sequence, with the same shape as
            `pose_sequence`.
        """
        mean = np.nanmean(pose_sequence, axis=(0, 1))
        std = np.nanstd(pose_sequence, axis=(0, 1))
        return np.divide(
            pose_sequence - mean, std, where=std != 0, out=np.zeros_like(pose_sequence)
        )


class MinMaxNormalization(Transform):
    """Rescales a pose sequence to the `[0, 1]` range per coordinate.

    For each coordinate channel (e.g. x, y, z), computes the min and max
    across all frames and landmarks (ignoring NaNs), and rescales the pose
    sequence so that channel falls within `[0, 1]`. Channels with zero
    range are left at zero instead of dividing by zero.

    Example:
        >>> import numpy as np
        >>> from sign_language_tools.pose.transform import MinMaxNormalization
        >>> pose_sequence = np.random.rand(10, 5, 2)  # (T, L, C)
        >>> transform = MinMaxNormalization()
        >>> normalized = transform(pose_sequence)
        >>> bool(normalized.min() >= 0 and normalized.max() <= 1)
        True
    """

    def __call__(self, pose_sequence: np.ndarray) -> np.ndarray:
        """Rescales the pose sequence to `[0, 1]`.

        Args:
            pose_sequence: Pose sequence of shape `(T, L, C)`, where `T` is
                the number of frames, `L` the number of landmarks, and `C`
                the number of coordinates per landmark.

        Returns:
            The rescaled pose sequence, with the same shape as
            `pose_sequence`.
        """
        min_val = np.nanmin(pose_sequence, axis=(0, 1))
        max_val = np.nanmax(pose_sequence, axis=(0, 1))
        scaling_factors = (max_val - min_val).reshape(1, 1, -1)
        return np.divide(
            pose_sequence - min_val,
            scaling_factors,
            where=scaling_factors != 0,
            out=np.zeros_like(pose_sequence),
        )


class FixedResolutionNormalization(Transform):
    """Normalizes a pose sequence expressed in pixel coordinates to `[-1, 1]`.

    Divides x and y coordinates by a fixed `(width, height)` resolution,
    then rescales the result from `[0, 1]` to `[-1, 1]`. Useful when
    landmarks were extracted from images/videos of a known, fixed size.

    Args:
        width: Width (in pixels) used to normalize the x-coordinate.
        height: Height (in pixels) used to normalize the y-coordinate.

    Example:
        >>> import numpy as np
        >>> from sign_language_tools.pose.transform import FixedResolutionNormalization
        >>> pose_sequence = np.array([[[0.0, 0.0], [640.0, 480.0]]])  # (T, L, C)
        >>> transform = FixedResolutionNormalization(width=640, height=480)
        >>> transform(pose_sequence).tolist()
        [[[-1.0, -1.0], [1.0, 1.0]]]
    """

    def __init__(self, width: int, height: int):
        super().__init__()
        self.resolution = np.array([width, height], dtype=np.float32)
        assert width > 0 and height > 0, "Both width and height must be greater than 0."

    def __call__(self, pose_sequence: np.ndarray) -> np.ndarray:
        """Normalizes the pose sequence to `[-1, 1]`.

        Args:
            pose_sequence: Pose sequence of shape `(T, L, 2)`, where `T` is
                the number of frames, `L` the number of landmarks, and the
                last dimension holds `(x, y)` pixel coordinates.

        Returns:
            The normalized pose sequence, with the same shape as
            `pose_sequence`.
        """
        return 2 * ((pose_sequence / self.resolution) - 0.5)
