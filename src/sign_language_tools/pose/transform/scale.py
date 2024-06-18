import numpy as np

from sign_language_tools.core.transform import Transform


class Scale(Transform):
    def __init__(self, scaling_factor: float, center=(0.5, 0.5)):
        super().__init__()
        self.center = center
        self.dx = (scaling_factor * center[0]) - center[0]
        self.dy = (scaling_factor * center[1]) - center[1]
        self.scaling_factor = scaling_factor

    def __call__(self, landmarks: np.ndarray):
        landmarks[:, :, :] *= self.scaling_factor
        landmarks[:, :, 0] -= self.dx
        landmarks[:, :, 1] -= self.dy
        return landmarks


class RandomScale(Scale):
    def __init__(self, min_scale: float, max_scale: float, center=(0.5, 0.5)):
        super().__init__(1.0, center)
        self.min_scale = min_scale
        self.max_scale = max_scale

    def __call__(self, landmarks: np.ndarray):
        scaling_factor = np.random.uniform(self.min_scale, self.max_scale)
        self.dx = (scaling_factor * self.center[0]) - self.center[0]
        self.dy = (scaling_factor * self.center[1]) - self.center[1]
        self.scaling_factor = scaling_factor
        return super().__call__(landmarks)

