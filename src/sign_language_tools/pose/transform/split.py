import numpy as np

from sign_language_tools.core.transform import Transform


class Split(Transform):
    def __init__(self, groups: dict[str, tuple[int, int]]):
        super().__init__()
        self.groups = groups

    def __call__(self, landmarks: np.ndarray):
        return {
            k: landmarks[:, start:end] for (k, (start, end)) in self.groups.items()
        }
