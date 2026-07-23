import numpy as np

from sign_language_tools.core.transform import Transform


class Split(Transform):
    """Splits a pose sequence into named groups of landmarks.

    Args:
        groups: Mapping from a group name to a `(start, end)` range of
            landmark indices (end exclusive) to include in that group.

    Example:
        >>> import numpy as np
        >>> from sign_language_tools.pose.transform import Split
        >>> pose_sequence = np.random.rand(10, 5, 2)  # (T, L, C)
        >>> transform = Split(groups={"left": (0, 2), "right": (2, 5)})
        >>> result = transform(pose_sequence)
        >>> result["left"].shape
        (10, 2, 2)
        >>> result["right"].shape
        (10, 3, 2)
    """

    def __init__(self, groups: dict[str, tuple[int, int]]):
        super().__init__()
        self.groups = groups

    def __call__(self, pose_sequence: np.ndarray) -> dict[str, np.ndarray]:
        """Splits the pose sequence into the configured groups.

        Args:
            pose_sequence: Pose sequence of shape `(T, L, C)`, where `T` is
                the number of frames, `L` the number of landmarks, and `C`
                the number of coordinates per landmark.

        Returns:
            A dict mapping each group name to its corresponding sub-sequence
            of landmarks, of shape `(T, end - start, C)`.
        """
        return {
            k: pose_sequence[:, start:end] for (k, (start, end)) in self.groups.items()
        }
