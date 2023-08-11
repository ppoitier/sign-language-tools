import numpy as np

from sign_language_tools.core.transform import Transform


class Flatten(Transform):

    def __call__(self, landmarks: np.ndarray):
        return landmarks.reshape(*landmarks.shape[:-2], -1)
