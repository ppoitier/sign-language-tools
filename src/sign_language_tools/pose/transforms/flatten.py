import numpy as np

from sign_language_tools.core.transform import Transform


class Flatten(Transform):
    """Flattens the last two dimensions of a pose sequence.

    Args:
        mode: Either `"landmarks"` (default), which merges the landmark
            and coordinate axes into one per frame (`(T, L, C) -> (T, L*C)`),
            or `"features"`, which merges the frame and coordinate axes
            into one per landmark (`(T, L, C) -> (L, T*C)`).

    Example:
        >>> import numpy as np
        >>> from sign_language_tools.pose.transform import Flatten
        >>> pose_sequence = np.random.rand(10, 5, 3)  # (T, L, C)
        >>> Flatten(mode="landmarks")(pose_sequence).shape
        (10, 15)
        >>> Flatten(mode="features")(pose_sequence).shape
        (5, 30)
    """

    def __init__(self, mode: str = "landmarks"):
        super().__init__()
        if mode not in ("landmarks", "features"):
            raise ValueError(f"mode must be 'landmarks' or 'features', got '{mode}'")
        self.mode = mode

    def __call__(self, pose_sequence: np.ndarray) -> np.ndarray:
        """Flattens the pose sequence.

        Args:
            pose_sequence: Pose sequence of shape `(T, L, C)`, where `T` is
                the number of frames, `L` the number of landmarks, and `C`
                the number of coordinates per landmark.

        Returns:
            The flattened pose sequence, of shape `(T, L*C)` if
            `mode="landmarks"`, or `(L, T*C)` if `mode="features"`.
        """
        if self.mode == "features":
            pose_sequence = pose_sequence.transpose((1, 0, 2))
        return pose_sequence.reshape(*pose_sequence.shape[:-2], -1)


class Unflatten(Transform):
    """Reshapes a flattened pose sequence back to `(T, L, C)`.

    Inverts a `Flatten` operation performed with the same `mode`,
    `n_landmarks`, and `n_channels`.

    Args:
        n_landmarks: Number of landmarks `L`.
        n_channels: Number of coordinates per landmark `C`, typically 3
            for (x, y, z).
        mode: Must match the mode used by `Flatten`. If `"features"`, a
            transpose is applied to restore the original `(T, L, C)`
            layout.

    Example:
        >>> import numpy as np
        >>> from sign_language_tools.pose.transform import Flatten, Unflatten
        >>> pose_sequence = np.random.rand(10, 5, 3)  # (T, L, C)
        >>> flat = Flatten(mode="landmarks")(pose_sequence)
        >>> Unflatten(n_landmarks=5, n_channels=3)(flat).shape
        (10, 5, 3)
    """

    def __init__(
        self,
        n_landmarks: int,
        n_channels: int,
        mode: str = "landmarks",
    ):
        super().__init__()
        if mode not in ("landmarks", "features"):
            raise ValueError(f"mode must be 'landmarks' or 'features', got '{mode}'")
        self.n_landmarks = n_landmarks
        self.n_channels = n_channels
        self.mode = mode

    def __call__(self, flat_pose_sequence: np.ndarray) -> np.ndarray:
        """Unflattens the pose sequence.

        Args:
            flat_pose_sequence: Flattened pose sequence, of shape
                `(T, L*C)` if `mode="landmarks"`, or `(L, T*C)` if
                `mode="features"`.

        Returns:
            The pose sequence reshaped back to `(T, L, C)`.
        """
        if self.mode == "landmarks":
            return flat_pose_sequence.reshape(-1, self.n_landmarks, self.n_channels)
        else:
            # Input is (L, T*C) → reshape to (L, T, C) → transpose to (T, L, C)
            n_frames = flat_pose_sequence.shape[-1] // self.n_channels
            return flat_pose_sequence.reshape(
                -1, n_frames, self.n_channels
            ).transpose((1, 0, 2))