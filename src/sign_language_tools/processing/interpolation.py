from typing import Callable

import numpy as np
from scipy.interpolate import interp1d


def get_landmark_interpolation_function(landmarks: np.ndarray, method: str = 'linear') -> Callable:
    """Return an approximation of the function f(t) = Y_t where t is any instant and Y the landmarks
    corresponding to it. For any given t, the function f uses interpolation to compute the resulting Y_t
    given the observed data.

    Example:
        ```
        # We have landmarks of shape (T, N, D)
        f = get_landmark_interpolation_function(landmarks)
        interpolated_landmark = f(7.32)
        ```

    Args:
        landmarks: Observed landmarks used to compute the interpolation when needed.
        method: Specify the method used to compute the interpolated landmarks:
            - `linear`
            - `nearest`
            - `previous`
            - `next`
    Returns:
        interp_func: the interpolation function

    Author:
        ppoitier (v1 03.04.2023)
    """
    # landmarks shape (T, N, C)
    t = landmarks.shape[0]
    x = np.argwhere(~np.isnan(landmarks.reshape(t, -1)).any(axis=1)).reshape(-1)
    y = landmarks[x]

    if len(y) < 2:
        method = 'nearest'

    return interp1d(
        x,
        y,
        kind=method,
        axis=0,
        assume_sorted=True,
        bounds_error=False,
        fill_value='extrapolate',
    )


def fill_empty_landmark_sequences(landmarks: np.ndarray, fill_values: float):
    landmarks[:, np.isnan(landmarks).all(axis=0).all(axis=1), :] = fill_values
    return landmarks


def interpolate_missing_landmarks(landmarks: np.ndarray, method: str = 'linear') -> np.ndarray:
    """
    Compute missing landmarks (NaN coordinates) using an interpolation function.
    See `get_landmark_interpolation_function` for more information about the interpolation.

    Args:
        landmarks: Tensor containing observed and missing landmarks.
        method: Specify the method used to compute the interpolated landmarks.

    Returns:
        interpolated_landmarks: Tensor containing observed and interpolated landmarks.

    Author:
        ppoitier (v1 03.04.2023)
    """
    f = get_landmark_interpolation_function(landmarks, method)
    return f(np.arange(landmarks.shape[0]))


