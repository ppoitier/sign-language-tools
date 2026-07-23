import numpy as np

from sign_language_tools.core.transform import Transform


class MoveSegments(Transform):
    """Shifts segments in time by a fixed offset.

    Only the start and end columns are shifted; any extra column (e.g. a
    per-segment label) is left untouched.

    Args:
        dx: Offset added to the start and end of every segment. Negative
            values shift segments earlier.

    Example:
        >>> import numpy as np
        >>> from sign_language_tools.annotations.transforms import MoveSegments
        >>> segments = np.array([[10, 20, 1], [30, 50, 2]])  # (M, 3)
        >>> transform = MoveSegments(dx=5)
        >>> transform(segments)
        array([[15, 25,  1],
               [35, 55,  2]])
    """

    def __init__(self, dx: float = 0.0):
        super().__init__()
        self.dx = dx

    def __call__(self, segments: np.ndarray) -> np.ndarray:
        """Shifts the segments.

        Args:
            segments: Array of shape `(M, 2)` or `(M, 3)` containing the
                start and end (and optionally a label) of `M` segments.

        Returns:
            The shifted segments, with the same shape as `segments`.
        """
        moved = segments.copy()
        moved[:, :2] = moved[:, :2] + self.dx
        return moved


class RandomRelativeMoveSegments(Transform):
    """Shifts segments in time by a random offset proportional to their length.

    Each segment is shifted independently by `dx_std * N(0, 1) * length`,
    where `length` is that segment's duration. Only the start and end
    columns are shifted; any extra column (e.g. a per-segment label) is
    left untouched.

    Args:
        dx_std: Standard deviation of the random shift, expressed as a
            fraction of each segment's length.

    Example:
        >>> import numpy as np
        >>> from sign_language_tools.annotations.transforms import RandomRelativeMoveSegments
        >>> segments = np.array([[10, 20, 1], [30, 50, 2]])  # (M, 3)
        >>> transform = RandomRelativeMoveSegments(dx_std=0.4)
        >>> moved = transform(segments)
        >>> moved.shape
        (2, 3)
    """

    def __init__(self, dx_std: float = 0.4):
        super().__init__()
        self.std = dx_std

    def __call__(self, segments: np.ndarray) -> np.ndarray:
        """Shifts the segments.

        Args:
            segments: Array of shape `(M, 2)` or `(M, 3)` containing the
                start and end (and optionally a label) of `M` segments.

        Returns:
            The shifted segments, with the same shape and dtype as
            `segments`.
        """
        moved = segments.copy()
        lengths = segments[:, 1] - segments[:, 0]
        dx = (self.std * np.random.randn(lengths.shape[0])) * lengths
        moved[:, :2] = np.round(moved[:, :2] + dx[:, None]).astype(segments.dtype)
        return moved
