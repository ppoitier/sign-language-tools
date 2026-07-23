import numpy as np

from sign_language_tools.core.transform import Transform


class TemporalCrop(Transform):
    """Crops a pose sequence to a fixed number of frames.

    If the pose sequence is already shorter than or equal to `size`, it is
    returned unchanged.

    Args:
        size: Number of frames to keep.
        location: Where to crop from: `"start"`, `"center"`, or `"end"`.

    Example:
        >>> import numpy as np
        >>> from sign_language_tools.pose.transform import TemporalCrop
        >>> pose_sequence = np.random.rand(10, 5, 2)  # (T, L, C)
        >>> transform = TemporalCrop(size=4, location="center")
        >>> transform(pose_sequence).shape
        (4, 5, 2)
    """

    def __init__(self, size: int, location: str = 'start'):
        super().__init__()
        self.size = size
        self.location = location

    def __call__(self, pose_sequence: np.ndarray) -> np.ndarray:
        """Crops the pose sequence.

        Args:
            pose_sequence: Pose sequence of shape `(T, L, C)`, where `T` is
                the number of frames, `L` the number of landmarks, and `C`
                the number of coordinates per landmark.

        Returns:
            The cropped pose sequence, of shape `(size, L, C)` (or
            unchanged if `T <= size`).
        """
        seq_len = pose_sequence.shape[0]
        if seq_len <= self.size:
            return pose_sequence
        if self.location == 'start':
            return pose_sequence[:self.size]
        elif self.location == 'center':
            start_idx = (pose_sequence.shape[0] - self.size) // 2
            return pose_sequence[start_idx:start_idx + self.size]
        elif self.location == 'end':
            return pose_sequence[-self.size:]
        else:
            raise ValueError(f"Unknown location: {self.location}. Please use 'start', 'center', or 'end'.")


class TemporalRandomCrop(Transform):
    """Crops a pose sequence to a fixed number of frames at a random offset.

    If the pose sequence is already shorter than or equal to `size`, it is
    returned unchanged. Otherwise, a random start position is chosen so
    that the whole crop fits within the sequence.

    Args:
        size: Number of frames to keep.

    Example:
        >>> import numpy as np
        >>> from sign_language_tools.pose.transform import TemporalRandomCrop
        >>> pose_sequence = np.random.rand(10, 5, 2)  # (T, L, C)
        >>> transform = TemporalRandomCrop(size=4)
        >>> transform(pose_sequence).shape
        (4, 5, 2)
    """

    def __init__(self, size: int):
        super().__init__()
        self.size = size

    def __call__(self, pose_sequence: np.ndarray) -> np.ndarray:
        """Crops the pose sequence at a random offset.

        Args:
            pose_sequence: Pose sequence of shape `(T, L, C)`, where `T` is
                the number of frames, `L` the number of landmarks, and `C`
                the number of coordinates per landmark.

        Returns:
            The cropped pose sequence, of shape `(size, L, C)` (or
            unchanged if `T <= size`).
        """
        T = pose_sequence.shape[0]
        if T <= self.size:
            return pose_sequence
        t = np.random.randint(low=0, high=T - self.size)
        return pose_sequence[t:t + self.size]
