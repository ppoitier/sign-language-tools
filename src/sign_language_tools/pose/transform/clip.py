import numpy as np

from sign_language_tools.core.transform import Transform


class Clip(Transform):
    def __init__(self, min_value: float = 0.0, max_value: float = 1.0):
        super().__init__()
        self.min = min_value
        self.max = max_value

    def __call__(self, landmarks: np.ndarray):
        return np.clip(landmarks, self.min, self.max)
