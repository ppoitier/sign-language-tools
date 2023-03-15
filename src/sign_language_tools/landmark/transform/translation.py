import numpy as np
import random


class Translation:
    def __init__(self, dx: float, dy: float):
        self.dx = dx
        self.dy = dy

    def __call__(self, landmarks: np.ndarray):
        landmarks[:, :, 0] += self.dx
        landmarks[:, :, 1] += self.dy
        return landmarks


class RandomTranslation(Translation):
    def __init__(self, dx_range=(-0.2, 0.2), dy_range=(-0.2, 0.2)):
        dx = random.uniform(*dx_range)
        dy = random.uniform(*dy_range)
        super().__init__(dx, dy)
