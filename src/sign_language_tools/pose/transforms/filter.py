import numpy as np

from sign_language_tools.core.transform import Transform


class FilterEmpty(Transform):
    """Removes frames where every landmark is missing.

    A frame is considered empty when all of its landmark coordinates are NaN
    (e.g. a frame where pose estimation failed on every landmark), and such
    frames are dropped from the sequence.

    Example:
        >>> import numpy as np
        >>> from sign_language_tools.pose.transform import FilterEmpty
        >>> pose_sequence = np.random.rand(3, 5, 2)  # (T, L, C)
        >>> pose_sequence[1] = np.nan  # frame 1 is entirely missing
        >>> transform = FilterEmpty()
        >>> transform(pose_sequence).shape
        (2, 5, 2)
    """

    def __call__(self, pose_sequence: np.ndarray) -> np.ndarray:
        """Removes empty frames from the pose sequence.

        Args:
            pose_sequence: Pose sequence of shape `(T, L, C)`, where `T` is
                the number of frames, `L` the number of landmarks, and `C`
                the number of coordinates per landmark.

        Returns:
            The pose sequence without its empty frames, of shape
            `(T', L, C)` with `T' <= T`.
        """
        return pose_sequence[~np.isnan(pose_sequence).all(axis=1).any(axis=1)]


class FilterLandmarks(Transform):
    """Removes a fixed set of landmarks from a pose sequence.

    Args:
        filtered_indices: Indices of the landmarks to remove.

    Example:
        >>> import numpy as np
        >>> from sign_language_tools.pose.transform import FilterLandmarks
        >>> pose_sequence = np.random.rand(10, 5, 2)  # (T, L, C)
        >>> transform = FilterLandmarks(filtered_indices=[0, 2])
        >>> transform(pose_sequence).shape
        (10, 3, 2)
    """

    def __init__(self, filtered_indices: set[int] | list[int]):
        super().__init__()
        self.filtered_indices = set(filtered_indices)

    def __call__(self, pose_sequence: np.ndarray) -> np.ndarray:
        """Removes the configured landmarks from the pose sequence.

        Args:
            pose_sequence: Pose sequence of shape `(T, L, C)`, where `T` is
                the number of frames, `L` the number of landmarks, and `C`
                the number of coordinates per landmark.

        Returns:
            The pose sequence without the filtered landmarks, of shape
            `(T, L - len(filtered_indices), C)`.
        """
        T, L, D = pose_sequence.shape
        lm_indices = [idx for idx in range(L) if idx not in self.filtered_indices]
        return pose_sequence[:, lm_indices]
