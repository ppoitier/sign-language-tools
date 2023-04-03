import numpy as np
from interpolation import get_landmark_interpolation_function


def resample(landmarks: np.ndarray, new_length: int, method: str = 'nearest'):
    # landmarks shape (T, N, C)
    t = landmarks.shape[0]
    x = np.linspace(0, t, new_length)
    f = get_landmark_interpolation_function(landmarks, method)
    return f(x)
