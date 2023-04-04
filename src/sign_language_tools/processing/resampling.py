import numpy as np
from sign_language_tools.processing.interpolation import get_landmark_interpolation_function


def resample(landmarks: np.ndarray, new_length: int, method: str = 'nearest') -> np.ndarray:
    """Resample a sequence of landmarks to a given length.

    Interpolation is used to compute new landmarks. See `interpolation.get_landmark_interpolation_function`.

    Args:
        landmarks: The original sequence of landmarks.
        new_length: The new length to resample.
        method: The method used to interpolate landmarks.

    Returns:
        resampled_landmarks: The sequence of landmarks that has been resampled to `new_length`.
    """
    # landmarks shape (T, N, C)
    t = landmarks.shape[0]
    x = np.linspace(0, t-1, new_length)
    f = get_landmark_interpolation_function(landmarks, method)
    return f(x)
