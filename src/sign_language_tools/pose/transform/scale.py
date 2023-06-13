import numpy as np


class Scale:
    def __init__(self, scaling_factor: float, center=(0.5, 0.5)):
        self.dx = (scaling_factor * center[0]) - center[0]
        self.dy = (scaling_factor * center[1]) - center[1]
        self.scaling_factor = scaling_factor

    def __call__(self, landmarks: np.ndarray):
        landmarks[:, :, :] *= self.scaling_factor
        landmarks[:, :, 0] -= self.dx
        landmarks[:, :, 1] -= self.dy
        return landmarks

