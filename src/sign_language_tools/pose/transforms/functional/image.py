import numpy as np


def pose_sequence_to_img(pose_sequence: np.ndarray, normalize: bool = True) -> np.ndarray:
    """Converts a pose sequence into an image-like array.

    Args:
        pose_sequence: Pose sequence of shape `(T, L, C)`, where `T` is
            the number of frames, `L` the number of landmarks, and `C`
            the number of coordinates per landmark.
        normalize: If `True`, rescales each coordinate channel to the
            `[0, 255]` range (per-frame min/max) and casts the result to
            `uint8`.

    Returns:
        Image-like array of shape `(C, L, T)`.
    """
    if normalize:
        poses_max = pose_sequence.max(axis=-2, keepdims=True)
        poses_min = pose_sequence.min(axis=-2, keepdims=True)
        img = np.round(255 * (pose_sequence - poses_min) / (poses_max - poses_min)).astype('uint8')
    else:
        img = pose_sequence
    return img.transpose((2, 1, 0))


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    _poses = np.load("D:/data/sign-languages/bobsl/poses/pose/5085344787448740525_809582_870529.npy")
    img = pose_sequence_to_img(_poses)

    print(img.shape)

    plt.figure()
    plt.imshow(img.transpose((1, 2, 0))[:, :100])
    plt.show()
