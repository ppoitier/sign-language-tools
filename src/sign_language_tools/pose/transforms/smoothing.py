import numpy as np
from scipy.signal import savgol_filter

from sign_language_tools.core.transform import Transform


class SavitzkyGolayFiltering(Transform):
    """Smooth a pose sequence with a Savitzky-Golay filter, applied along the time axis.

    If the sequence is shorter than the filter window, it is returned unchanged.

    Args:
        window_length: Length of the filter window.
        polynom_order: Order of the polynomial used to fit each window.
    """

    def __init__(self, window_length: int, polynom_order: int):
        super().__init__()
        self.window_length = window_length
        self.polynom_order = polynom_order

    def __call__(self, pose_sequence: np.ndarray) -> np.ndarray:
        # pose_sequence shape: (T, L, C)
        if pose_sequence.shape[0] < self.window_length:
            return pose_sequence

        return savgol_filter(pose_sequence, self.window_length, self.polynom_order, axis=0)
