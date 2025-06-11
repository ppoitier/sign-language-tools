import numpy as np

from sign_language_tools.core.transform import Transform


class HorizontalFlip(Transform):
    def __init__(self, origin: tuple[float, float] = (0.5, 0.5)):
        super().__init__()
        self.origin_x = origin[0]

    def __call__(self, landmarks: np.ndarray):
        landmarks[:, :, 0] = -(landmarks[:, :, 0] - self.origin_x) + self.origin_x
        # landmarks[:, :, 0] = 1 - landmarks[:, :, 0]
        return landmarks
