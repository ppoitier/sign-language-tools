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


class Unflatten(Transform):
    def __init__(self, n_landmarks: int, n_coords: int):
        super().__init__()
        self.n_landmarks = n_landmarks
        self.n_coords = n_coords

    def __call__(self, flat_landmarks: np.ndarray) -> np.ndarray:
        return flat_landmarks.reshape(-1, self.n_landmarks, self.n_coords)
