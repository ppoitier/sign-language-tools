import numpy as np

from sign_language_tools.core.transform import Transform


class HorizontalFlip(Transform):

    def __call__(self, landmarks: np.ndarray):
        landmarks[:, :, 0] = 1 - landmarks[:, :, 0]
        return landmarks
