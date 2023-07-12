from typing import Union

import numpy as np

from sign_language_tools.core.transform import Transform


class Concatenate(Transform):
    def __init__(self, landmark_sets: list[str] = None, landmark_axis=1):
        super().__init__()
        self.landmark_axis = landmark_axis
        self.landmark_sets = landmark_sets

    def __call__(self, poses: Union[dict[any, np.ndarray], list[np.ndarray]]):
        if isinstance(poses, dict):
            if self.landmark_sets is None:
                self.landmark_sets = list(sorted(poses.keys()))
            poses = [poses[key] for key in self.landmark_sets]
        return np.concatenate(poses, axis=self.landmark_axis)
