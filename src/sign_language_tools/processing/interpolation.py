import numpy as np
from scipy.interpolate import interp1d


def get_landmark_interpolation_function(landmarks: np.ndarray, method: str = 'linear'):
    # landmarks shape (T, N, C)
    t = landmarks.shape[0]
    x = np.argwhere(~np.isnan(landmarks.reshape(t, -1)).any(axis=1)).reshape(-1)
    y = landmarks[x]
    return interp1d(
        x,
        y,
        kind=method,
        axis=0,
        assume_sorted=True,
    )


def interpolate_missing_landmarks(landmarks: np.ndarray, method: str = 'linear'):
    f = get_landmark_interpolation_function(landmarks, method)
    return f(np.arange(landmarks.shape[0]))


