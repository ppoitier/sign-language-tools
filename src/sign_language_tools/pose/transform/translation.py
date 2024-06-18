import numpy as np
import random

from sign_language_tools.core.transform import Transform


class Translation(Transform):
    def __init__(self, dx: float, dy: float):
        super().__init__()
        self.dx = dx
        self.dy = dy

    def __call__(self, landmarks: np.ndarray):
        landmarks[:, :, 0] += self.dx
        landmarks[:, :, 1] += self.dy
        return landmarks


class RandomTranslation(Transform):
    def __init__(self, dx_range=(-0.2, 0.2), dy_range=(-0.2, 0.2)):
        super().__init__()
        self.dx_range = dx_range
        self.dy_range = dy_range

    def __call__(self, landmarks: np.ndarray):
        landmarks[:, :, 0] += random.uniform(*self.dx_range)
        landmarks[:, :, 1] += random.uniform(*self.dy_range)
        return landmarks
