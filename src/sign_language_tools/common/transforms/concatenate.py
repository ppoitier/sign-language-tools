import numpy as np

from sign_language_tools.core.transform import Transform


class Concatenate(Transform):
    def __init__(self, dim: int = 0):
        super().__init__()
        self.dim = dim

    def __call__(self, x: tuple[np.ndarray]) -> np.ndarray:
        return np.concatenate(x, axis=self.dim)
