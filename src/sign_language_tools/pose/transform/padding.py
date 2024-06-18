import numpy as np

from sign_language_tools.core.transform import Transform


class Padding(Transform):
    def __init__(
        self,
        min_length: int,
        location="right",
        mode: str = "constant",
        constant_value: float = 0.0,
    ):
        super().__init__()
        self.min_length = min_length
        self.location = location
        self.mode = mode
        self.constant_value = constant_value

    def __call__(self, pose_sequence: np.ndarray) -> np.ndarray:
        T = pose_sequence.shape[0]

        if T >= self.min_length:
            return pose_sequence

        padding = self.min_length - T
        if self.location == "right":
            pad_width = ((0, padding), (0, 0), (0, 0))

        else:
            pad_width = ((padding, 0), (0, 0), (0, 0))

        if self.mode == 'constant':
            return np.pad(pose_sequence, pad_width, constant_values=self.constant_value)
        elif self.mode == 'repeat':
            n_repeats = int(np.ceil(self.min_length / T))
            return np.tile(pose_sequence, (n_repeats, 1, 1))[:self.min_length]
        else:
            return np.pad(pose_sequence, pad_width, mode=self.mode)
