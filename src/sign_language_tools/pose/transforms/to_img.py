import numpy as np

from sign_language_tools.core.transform import Transform
from sign_language_tools.pose.transforms.functional.image import pose_sequence_to_img


class ToRGBImage(Transform):
    """Converts a pose sequence into a 3-channel image-like array.

    Coordinates are rearranged into an image layout with landmarks as rows
    and frames as columns. If the pose sequence only has 2 coordinate
    channels (e.g. `(x, y)`), a third all-zero channel is appended so the
    result can be treated as an RGB image.

    Args:
        normalize: If `True`, rescales each coordinate channel to the
            `[0, 255]` range (per-frame min/max) and casts the result to
            `uint8`.
        fill_z_with_zero: If `True` and the pose sequence has only 2
            coordinate channels, appends a third all-zero channel.

    Example:
        >>> import numpy as np
        >>> from sign_language_tools.pose.transforms import ToRGBImage
        >>> pose_sequence = np.random.rand(10, 33, 2)  # (T, L, C)
        >>> transform = ToRGBImage()
        >>> img = transform(pose_sequence)
        >>> img.shape
        (3, 33, 10)
    """

    def __init__(
            self,
            normalize: bool = True,
            fill_z_with_zero: bool = True,
    ):
        super().__init__()
        self.normalize = normalize
        self.fill_z_with_zero = fill_z_with_zero

    def __call__(self, pose_sequence: np.ndarray) -> np.ndarray:
        """Converts the pose sequence into an image-like array.

        Args:
            pose_sequence: Pose sequence of shape `(T, L, C)`, where `T` is
                the number of frames, `L` the number of landmarks, and `C`
                the number of coordinates per landmark.

        Returns:
            Image-like array of shape `(C, L, T)`, with `C` equal to 3 if
            `fill_z_with_zero` is `True` and the input had 2 coordinate
            channels, otherwise unchanged.
        """
        x = pose_sequence_to_img(pose_sequence, normalize=self.normalize)
        if self.fill_z_with_zero and x.shape[0] == 2:
            x = np.concatenate([x, np.zeros_like(x[[0]])], axis=0)
        return x