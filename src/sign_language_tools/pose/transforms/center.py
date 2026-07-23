import numpy as np

from sign_language_tools.core.transform import Transform


class CenterOnLandmarks(Transform):
    """Centers a pose sequence on one or more reference landmarks.

    For every frame, the mean position of the given reference landmark(s) is
    subtracted from all landmarks, so that the reference point becomes the
    new origin.

    Args:
        landmark_idx: Index, or tuple of indices, of the landmark(s) used as
            the reference point. If several indices are given, their mean
            position is used as the origin.

    Example:
        >>> import numpy as np
        >>> from sign_language_tools.pose.transforms import CenterOnLandmarks
        >>> pose_sequence = np.random.rand(10, 33, 3)  # (T, L, C)
        >>> transform = CenterOnLandmarks(landmark_idx=0)
        >>> centered = transform(pose_sequence)
        >>> centered.shape
        (10, 33, 3)
    """

    def __init__(self, landmark_idx: int | tuple[int, ...]):
        super().__init__()
        if isinstance(landmark_idx, int):
            landmark_idx = (landmark_idx,)
        self.landmark_idx = landmark_idx

    def __call__(self, pose_sequence: np.ndarray) -> np.ndarray:
        """Centers the pose sequence.

        Args:
            pose_sequence: Pose sequence of shape `(T, L, C)`, where `T` is
                the number of frames, `L` the number of landmarks, and `C`
                the number of coordinates per landmark.

        Returns:
            The centered pose sequence, with the same shape as `pose_sequence`.
        """
        ref = pose_sequence[:, self.landmark_idx, :].mean(axis=1)[:, None]
        return pose_sequence - ref
