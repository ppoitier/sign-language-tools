from math import floor, ceil

import numpy as np


def moving_average(arr: np.ndarray, win_size: int, padded: bool = True) -> np.ndarray:
    if padded:
        padding = (win_size - 1) / 2
        arr = np.pad(arr, (floor(padding), ceil(padding)), mode='edge')
    return np.convolve(arr, np.ones(win_size), 'valid') / win_size


def max_pool1d(arr: np.ndarray, win_size: int, stride: int = 1, padded: bool = True):
    if padded:
        padding = (win_size - 1) / 2
        arr = np.pad(arr, (floor(padding), ceil(padding)), mode='edge')
    windows = np.lib.stride_tricks.as_strided(
        arr,
        shape=((arr.shape[0] - win_size) // stride + 1, win_size),
        strides=(arr.strides[0] * stride, arr.strides[0]),
    )
    return np.maximum.reduce(windows)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    x = np.linspace(-10, 10, 100)
    x += 0.2 * np.random.randn(100)
    x = x**2

    x_smoothed = moving_average(x, win_size=10, padded=True)

    plt.figure()
    plt.plot(x)
    plt.plot(x_smoothed)
    plt.show()

    plt.figure()
    plt.plot(x - x_smoothed)
    plt.show()
