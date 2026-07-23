from typing import Optional

import numpy as np

from sign_language_tools.core.transform import Transform


class DropRandomFrames(Transform):
    def __init__(
            self,
            max_n: Optional[int] = None,
            max_p: Optional[float] = None,
            fill_value: float = 0.0,
    ):
        super().__init__()
        assert (max_n is None) ^ (max_p is None), """
        You should specify either the maximum number of dropped frames
        or the maximum proportion of dropped frames.
        """
        assert (max_p is None) or (0 <= max_p <= 1)
        self.max_n = max_n
        self.max_p = max_p
        self.fill_value = fill_value

    def __call__(self, landmarks: np.ndarray) -> np.ndarray:
        landmarks = landmarks.copy()
        T = landmarks.shape[0]
        if self.max_n is not None:
            n = self.max_n
        else:
            n = self.max_p * T
        n = np.random.randint(low=0, high=n)
        dropped_frames = np.random.choice(T, size=n, replace=False)
        landmarks[dropped_frames] = self.fill_value
        return landmarks
