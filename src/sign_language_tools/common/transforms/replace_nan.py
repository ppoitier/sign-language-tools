import numpy as np

from sign_language_tools.core.transform import Transform


class ReplaceNaN(Transform):
    def __init__(self, fill_value: float = 0.0):
        super().__init__()
        self.fill_value = fill_value

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x[np.isnan(x)] = self.fill_value
        return x
