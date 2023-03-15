import numpy as np


class GaussianNoise:
    def __init__(self, scale: float):
        self.scale = scale

    def __call__(self, landmarks: np.ndarray):
        noise = np.random.normal(scale=self.scale, size=landmarks.shape)
        return landmarks + noise
