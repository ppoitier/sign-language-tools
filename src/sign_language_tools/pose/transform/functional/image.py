import numpy as np


def pose_sequence_to_img(poses: np.ndarray) -> np.ndarray:
    poses_max = poses.max(axis=-2, keepdims=True)
    poses_min = poses.min(axis=-2, keepdims=True)
    return np.round(255 * (poses - poses_min) / (poses_max - poses_min)).astype('uint8').transpose((1, 0, 2))


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    _poses = np.load("D:/data/sign-languages/bobsl/poses/pose/5085344787448740525_809582_870529.npy")
    img = pose_sequence_to_img(_poses)

    print(img.shape)

    plt.figure()
    plt.imshow(img[:, :100])
    plt.show()
