import numpy as np


class Padding:
    def __init__(
        self,
        min_length: int,
        location="right",
        mode: str = "constant",
        constant_value: float = 0.0,
    ):
        self.min_length = min_length
        self.location = location
        self.mode = mode
        self.constant_value = constant_value

    def __call__(self, pose_sequence: np.ndarray) -> np.ndarray:
        if pose_sequence.shape[0] >= self.min_length:
            return pose_sequence

        padding = self.min_length - pose_sequence.shape[0]
        if self.location == "right":
            pad_width = ((0, padding), (0, 0), (0, 0))

        else:
            pad_width = ((padding, 0), (0, 0), (0, 0))

        return np.pad(
            pose_sequence,
            pad_width,
            mode=self.mode,
            constant_values=self.constant_value,
        )
