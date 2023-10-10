import numpy as np
from scipy.signal import savgol_filter

from sign_language_tools.core.transform import Transform


class SavitchyGolayFiltering(Transform):
    """
    Smooth the landmarks by applying a Savitchy-Golay filter. If the window of the filter
    is bigger then the sequence, the sequence is returned unchanged.

    Args:
        window_length: Length of the filter window for Savitchy-Golay.µ
        polynom_order: Order of the polynom used by Savitchy-Golay.

    Returns:
        interpolated_landmarks: Tensor containing observed and interpolated landmarks.

    Author:
        jfink (v1.5 10.10.2023)
    """

    def __init__(self, window_length: int, polynom_order: int):
        super().__init__()
        self.window_length = window_length
        self.polynom_order = polynom_order

    def __call__(self, pose_sequence: np.ndarray):
        if pose_sequence.shape[0] < self.window_length:
            return pose_sequence

        return np.apply_along_axis(
            lambda seq: savgol_filter(seq, self.window_length, self.polynom_order),
            axis=0,
            arr=pose_sequence,
        )
