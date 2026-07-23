from typing import Literal

import numpy as np

from sign_language_tools.core.transform import Transform


class ScaleSegments(Transform):
    """Scales the length of segments by a fixed factor.

    Each segment's length is multiplied by `factor` and clipped to
    `[min_length, max_length]`. The resized segment is then repositioned
    according to `location`, so that either its center, its start, or its
    end stays fixed.

    Args:
        factor: Multiplicative factor applied to each segment's length.
            Values below 1.0 shrink segments, values above 1.0 grow them.
        location: Which point of the segment stays fixed while resizing:
            `"center"` keeps the midpoint fixed, `"start"` keeps the start
            fixed, and `"end"` keeps the end fixed.
        min_length: Lower bound applied to the new segment length after
            scaling.
        max_length: Upper bound applied to the new segment length after
            scaling.

    Note:
        If `segments` has an integer dtype, the repositioned start/end
        values are truncated (not rounded) to that dtype. Pass a float
        array if fractional precision matters.

    Example:
        >>> import numpy as np
        >>> from sign_language_tools.annotations.transforms import ScaleSegments
        >>> segments = np.array([[10., 20.], [30., 60.]])  # (M, 2)
        >>> transform = ScaleSegments(factor=0.5, location="center")
        >>> transform(segments)
        array([[12.5, 17.5],
               [37.5, 52.5]])
    """

    def __init__(
        self,
        factor: float = 0.8,
        location: Literal["center", "start", "end"] = "center",
        min_length: float = 1.0,
        max_length: float = 1000.0,
    ):
        super().__init__()
        self.factor = factor
        self.location = location
        self.min_length = min_length
        self.max_length = max_length

    def _compute_new_lengths(self, lengths: np.ndarray) -> np.ndarray:
        return lengths * self.factor

    def _resize_segments(self, segments: np.ndarray, lengths: np.ndarray, new_lengths: np.ndarray) -> np.ndarray:
        if self.location == "center":
            centers = segments[:, 0] + lengths / 2
            segments[:, 0] = centers - new_lengths / 2
            segments[:, 1] = centers + new_lengths / 2
        elif self.location == "start":
            segments[:, 1] = segments[:, 0] + new_lengths
        elif self.location == "end":
            segments[:, 0] = segments[:, 1] - new_lengths
        else:
            raise ValueError(f"Unknown location: '{self.location}'")
        return segments

    def __call__(self, segments: np.ndarray) -> np.ndarray:
        """Scales the segments.

        Args:
            segments: Array of shape `(M, 2)` containing the start and end
                of `M` segments.

        Returns:
            The scaled segments, with the same shape as `segments`.
        """
        if segments.shape[1] != 2:
            raise ValueError("Segments must have shape (M, 2)")
        scaled_segments = segments.copy()
        lengths = segments[:, 1] - segments[:, 0]
        new_lengths = np.clip(self._compute_new_lengths(lengths), self.min_length, self.max_length)
        return self._resize_segments(scaled_segments, lengths, new_lengths)


class RandomRelativeScaleSegments(ScaleSegments):
    """Scales the length of segments by a random factor drawn per segment.

    Each segment's length is multiplied by an independent factor sampled as
    `1 + scale_std * N(0, 1)`, then clipped to `[min_length, max_length]` and
    repositioned according to `location`, exactly like `ScaleSegments`.

    Args:
        scale_std: Standard deviation of the random scaling factor around 1.0.
        location: Which point of the segment stays fixed while resizing:
            `"center"` keeps the midpoint fixed, `"start"` keeps the start
            fixed, and `"end"` keeps the end fixed.
        min_length: Lower bound applied to the new segment length after
            scaling.
        max_length: Upper bound applied to the new segment length after
            scaling.

    Example:
        >>> import numpy as np
        >>> from sign_language_tools.annotations.transforms import RandomRelativeScaleSegments
        >>> segments = np.array([[10., 20.], [30., 60.]])  # (M, 2)
        >>> transform = RandomRelativeScaleSegments(scale_std=0.3)
        >>> scaled = transform(segments)
        >>> scaled.shape
        (2, 2)
    """

    def __init__(
        self,
        scale_std: float = 0.5,
        location: Literal["center", "start", "end"] = "center",
        min_length: float = 1.0,
        max_length: float = 1000.0,
    ):
        super().__init__(factor=1.0, location=location, min_length=min_length, max_length=max_length)
        self.std = scale_std

    def _compute_new_lengths(self, lengths: np.ndarray) -> np.ndarray:
        return (self.std * np.random.randn(lengths.shape[0]) + 1) * lengths
