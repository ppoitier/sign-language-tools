import numpy as np

from sign_language_tools.core.transform import Transform


class MoveSegments(Transform):
    def __init__(self, dx: float = 0.0):
        super().__init__()
        self.dx = dx

    def __call__(self, segments: np.ndarray) -> np.ndarray:
        return segments + self.dx


class RandomRelativeMoveSegments(Transform):
    def __init__(self, dx_std: float = 0.4):
        super().__init__()
        self.std = dx_std

    def __call__(self, segments: np.ndarray) -> np.ndarray:
        lengths = segments[:, 1] - segments[:, 0]
        dx = (self.std * np.random.randn(lengths.shape[0])) * lengths
        return segments + dx[:, None]
