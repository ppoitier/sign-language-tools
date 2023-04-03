import numpy as np


class HorizontalFlip:

    def __call__(self, landmarks: np.ndarray):
        landmarks[:, :, 0] = 1 - landmarks[:, :, 0]
        return landmarks
