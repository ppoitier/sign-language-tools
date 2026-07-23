import numpy as np

from sign_language_tools.core.transform import Transform


class Padding(Transform):
    def __init__(
        self,
        min_length: int,
        location="right",
        mode: str = "constant",
        constant_value: float = 0.0,
        return_mask: bool = False,
    ):
        super().__init__()
        self.min_length = min_length
        self.location = location
        self.mode = mode
        self.constant_value = constant_value
        self.return_mask = return_mask

    def __call__(self, pose_sequence: np.ndarray) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        T = pose_sequence.shape[0]

        if T >= self.min_length:
            if self.return_mask:
                return pose_sequence, np.ones((T,), dtype='uint8')
            else:
                return pose_sequence

        padding = self.min_length - T
        if self.location == "right":
            pad_width = ((0, padding),)
        else:
            pad_width = ((padding, 0),)
        pad_width += ((0, 0),) * (len(pose_sequence.shape)-1)

        if self.mode == 'constant':
            # noinspection PyTypeChecker
            padded_pose_sequence = np.pad(pose_sequence, pad_width, constant_values=self.constant_value)
        elif self.mode == 'repeat':
            n_repeats = int(np.ceil(self.min_length / T))
            padded_pose_sequence = np.tile(pose_sequence, (n_repeats, 1, 1))[:self.min_length]
        else:
            # noinspection PyTypeChecker
            padded_pose_sequence = np.pad(pose_sequence, pad_width, mode=self.mode)

        if self.return_mask:
            mask = np.ones((self.min_length,), dtype='uint8')
            if self.location == "right":
                mask[-padding:] = 0
            else:
                mask[:padding] = 0
            return padded_pose_sequence, mask

        return padded_pose_sequence
