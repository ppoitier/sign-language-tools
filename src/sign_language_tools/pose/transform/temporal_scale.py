import math
import random

import numpy as np

from sign_language_tools.pose.transform.resample import Resample


class TemporalScale(Resample):
    def __init__(self, scale: float, time_axis=0):
        super().__init__(new_length=0)
        self.scale = scale
        self.time_axis = time_axis

    def __call__(self, pose_sequence: np.ndarray) -> np.ndarray:
        self.new_length = math.ceil(pose_sequence.shape[self.time_axis] * self.scale)
        return super().__call__(pose_sequence)


class RandomTemporalScale(Resample):
    def __init__(self, min_scale: float, max_scale: float, time_axis=0):
        super().__init__(new_length=0)
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.time_axis = time_axis

    def __call__(self, pose_sequence: np.ndarray) -> np.ndarray:
        scale = random.uniform(self.min_scale, self.max_scale)
        self.new_length = math.ceil(pose_sequence.shape[self.time_axis] * scale)
        return super().__call__(pose_sequence)

