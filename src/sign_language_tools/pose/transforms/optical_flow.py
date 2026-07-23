import numpy as np

from sign_language_tools.core.transform import Transform


class ToOpticalFlow(Transform):
    """Computes the per-landmark displacement between consecutive frames.

    For each frame and each landmark, computes the Euclidean distance
    between its position in the current and previous frame (the first
    frame is compared against the origin). This yields a per-landmark
    optical-flow-like signal that is high when a landmark moves quickly.

    Args:
        fps: If given, multiplies the displacement by this frame rate to
            convert it from a per-frame value into a velocity (per second).

    Example:
        >>> import numpy as np
        >>> from sign_language_tools.pose.transform import ToOpticalFlow
        >>> pose_sequence = np.random.rand(10, 5, 2)  # (T, L, C)
        >>> transform = ToOpticalFlow()
        >>> transform(pose_sequence).shape
        (10, 5)
    """

    def __init__(self, fps: float | None = None):
        super().__init__()
        self.fps = fps

    def __call__(self, pose_sequence: np.ndarray) -> np.ndarray:
        """Computes the optical flow of the pose sequence.

        Args:
            pose_sequence: Pose sequence of shape `(T, L, C)`, where `T` is
                the number of frames, `L` the number of landmarks, and `C`
                the number of coordinates per landmark.

        Returns:
            Array of shape `(T, L)` containing, for each frame and
            landmark, the displacement (or velocity if `fps` is set) since
            the previous frame.
        """
        optical_flow = np.nan_to_num(np.linalg.norm(np.diff(pose_sequence, axis=0, prepend=0), axis=-1))
        if self.fps is not None:
            optical_flow *= self.fps
        return optical_flow
