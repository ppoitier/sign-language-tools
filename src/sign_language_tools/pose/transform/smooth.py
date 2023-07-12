import numpy as np
from scipy.signal import savgol_filter

from sign_language_tools.core.transform import Transform


class SavitchyGolayFiltering(Transform):
    def __init__(self, window_length: int, polynom_order: int):
        super().__init__()
        self.window_length = window_length
        self.polynom_order = polynom_order

    def __call__(self, pose_sequence: np.ndarray):
        return np.apply_along_axis(
            lambda seq: savgol_filter(seq, self.window_length, self.polynom_order),
            axis=0,
            arr=pose_sequence,
        )
