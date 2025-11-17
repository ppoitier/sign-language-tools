import numpy as np

from sign_language_tools.core.transform import Transform
from sign_language_tools.pose.transform.functional.image import pose_sequence_to_img


class ToRGBImage(Transform):
    def __init__(
            self,
            normalize_landmarks_signals: bool = False,
            fill_z_with_zero: bool = True,
    ):
        super().__init__()
        self.normalize_landmarks_signals = normalize_landmarks_signals
        self.fill_z_with_zero = fill_z_with_zero

    def __call__(self, img):
        x = pose_sequence_to_img(img)
        if self.normalize_landmarks_signals:
            ... # TODO
        if self.fill_z_with_zero and x.shape[-1] == 2:
            x = np.concatenate([x, np.zeros_like(x[:, :, [0]])], axis=-1)
        return x
