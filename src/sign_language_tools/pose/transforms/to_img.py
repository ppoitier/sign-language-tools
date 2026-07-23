import numpy as np

from sign_language_tools.core.transform import Transform
from sign_language_tools.pose.transforms.functional.image import pose_sequence_to_img


class ToRGBImage(Transform):
    def __init__(
            self,
            normalize: bool = True,
            fill_z_with_zero: bool = True,
    ):
        super().__init__()
        self.normalize = normalize
        self.fill_z_with_zero = fill_z_with_zero

    def __call__(self, pose_sequence):
        x = pose_sequence_to_img(pose_sequence, normalize=self.normalize)
        if self.fill_z_with_zero and x.shape[-1] == 2:
            x = np.concatenate([x, np.zeros_like(x[:, :, [0]])], axis=-1)
        return x
