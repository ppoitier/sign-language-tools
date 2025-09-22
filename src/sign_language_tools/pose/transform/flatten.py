import numpy as np

from sign_language_tools.core.transform import Transform


class Flatten(Transform):
    def __init__(self, item: str = 'landmarks'):
        super().__init__()
        assert item in ('landmarks', 'features')
        self.item = item

    def __call__(self, landmarks: np.ndarray):
        if self.item == 'features':
            landmarks = landmarks.transpose((1, 0, 2))
        return landmarks.reshape(*landmarks.shape[:-2], -1)
