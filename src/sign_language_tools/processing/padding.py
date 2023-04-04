import numpy as np


def pad(landmarks: np.ndarray, new_length: int, mode: str = 'constant', fill_value: float = -1.):
    padding = max(0, new_length - landmarks.shape[0])
    if mode == 'constant':
        return np.pad(landmarks, ((0, padding), (0, 0), (0, 0)), mode=mode, constant_values=fill_value)
    else:
        return np.pad(landmarks, ((0, padding), (0, 0), (0, 0)), mode=mode)
