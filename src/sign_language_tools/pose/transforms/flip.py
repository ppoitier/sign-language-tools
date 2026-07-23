import numpy as np

from sign_language_tools.core.transform import Transform


class HorizontalFlip(Transform):
    def __init__(self, x_origin: float = 0.5):
        super().__init__()
        self.x_origin = x_origin

    def __call__(self, landmarks: np.ndarray):
        landmarks = landmarks.copy()
        landmarks[:, :, 0] = -(landmarks[:, :, 0] - self.x_origin) + self.x_origin
        return landmarks
