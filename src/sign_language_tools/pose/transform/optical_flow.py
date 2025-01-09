import numpy as np

from sign_language_tools.core.transform import Transform


class ToOpticalFlow(Transform):
    def __init__(self, fps: float | None = None):
        super().__init__()
        self.fps = fps

    def __call__(self, poses: np.ndarray) -> np.ndarray:
        optical_flow = np.nan_to_num(np.linalg.norm(np.diff(poses, axis=0, prepend=0), axis=-1))
        if self.fps is not None:
            optical_flow *= self.fps
        return optical_flow
