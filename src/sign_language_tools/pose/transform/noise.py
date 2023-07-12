import numpy as np

from sign_language_tools.core.transform import Transform


class GaussianNoise(Transform):
    def __init__(self, scale: float):
        super().__init__()
        self.scale = scale

    def __call__(self, landmarks: np.ndarray):
        noise = np.random.normal(scale=self.scale, size=landmarks.shape)
        return landmarks + noise
