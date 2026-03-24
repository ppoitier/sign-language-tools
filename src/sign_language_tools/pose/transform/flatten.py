import numpy as np

from sign_language_tools.core.transform import Transform


class Flatten(Transform):
    """Flatten the last two dimensions of pose data.

    Given an input of shape (T, L, C), produces:
    - (T, L*C) when mode='landmarks' (merge landmarks and coordinates per frame)
    - (L, T*C) when mode='features' (merge frames and coordinates per landmark)

    Args:
        mode: Either 'landmarks' (default) or 'features'.
    """

    def __init__(self, mode: str = "landmarks"):
        super().__init__()
        if mode not in ("landmarks", "features"):
            raise ValueError(f"mode must be 'landmarks' or 'features', got '{mode}'")
        self.mode = mode

    def __call__(self, landmarks: np.ndarray) -> np.ndarray:
        if self.mode == "features":
            landmarks = landmarks.transpose((1, 0, 2))
        return landmarks.reshape(*landmarks.shape[:-2], -1)


class Unflatten(Transform):
    """Unflatten pose data back to (..., L, C) shape.

    Inverts a Flatten(mode='landmarks') operation by reshaping the last
    dimension from L*C back to (L, C).

    Args:
        n_landmarks: Number of landmarks (L).
        n_channels: Number of channels per landmark (C), typically 3 for XYZ.
        mode: Must match the mode used during flattening. If 'features',
            a transpose is applied to restore the original (T, L, C) layout.
    """

    def __init__(
        self,
        n_landmarks: int,
        n_channels: int,
        mode: str = "landmarks",
    ):
        super().__init__()
        if mode not in ("landmarks", "features"):
            raise ValueError(f"mode must be 'landmarks' or 'features', got '{mode}'")
        self.n_landmarks = n_landmarks
        self.n_channels = n_channels
        self.mode = mode

    def __call__(self, flat_landmarks: np.ndarray) -> np.ndarray:
        if self.mode == "landmarks":
            return flat_landmarks.reshape(-1, self.n_landmarks, self.n_channels)
        else:
            # Input is (L, T*C) → reshape to (L, T, C) → transpose to (T, L, C)
            n_frames = flat_landmarks.shape[-1] // self.n_channels
            return flat_landmarks.reshape(
                -1, n_frames, self.n_channels
            ).transpose((1, 0, 2))