import numpy as np

from sign_language_tools.core.transform import Transform


class CenterOnLandmarks(Transform):
    def __init__(self, landmark_idx: int | tuple[int, ...]):
        super().__init__()
        if isinstance(landmark_idx, int):
            landmark_idx = (landmark_idx,)
        self.landmark_idx = landmark_idx

    def __call__(self, pose_seq: np.ndarray) -> np.ndarray:
        ref = pose_seq[:, self.landmark_idx, :].mean(axis=1)[:, None]
        return pose_seq - ref
