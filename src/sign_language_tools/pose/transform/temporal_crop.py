import numpy as np

from sign_language_tools.core.transform import Transform


class TemporalRandomCrop(Transform):
    def __init__(self, size: int):
        super().__init__()
        self.size = size

    def __call__(self, landmarks: np.ndarray):
        T = landmarks.shape[0]
        if T <= self.size:
            return landmarks
        t = np.random.randint(low=0, high=T - self.size)
        return landmarks[t:t + self.size]
