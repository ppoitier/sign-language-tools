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


class RandomTranslation(Translation):
    def __init__(self, dx_range=(-0.2, 0.2), dy_range=(-0.2, 0.2)):
        self.dx_range = dx_range
        self.dy_range = dy_range

        dx = random.uniform(*dx_range)
        dy = random.uniform(*dy_range)
        super().__init__(dx, dy)

    def __call__(self, landmarks: np.ndarray):
        res = super().__call__(landmarks)
        self.dx = random.uniform(*self.dx_range)
        self.dy = random.uniform(*self.dy_range)

        return res
