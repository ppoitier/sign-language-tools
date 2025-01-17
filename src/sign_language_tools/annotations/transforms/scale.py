from typing import Literal

import numpy as np

from sign_language_tools.core.transform import Transform


class ScaleSegments(Transform):
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

    def _resize_segments(self, segments: np.ndarray, lengths: np.ndarray, new_lengths: np.ndarray) -> np.ndarray:
        if self.location == "center":
            centers = segments[:, 0] + lengths / 2
            segments[:, 0] = centers - new_lengths / 2
            segments[:, 1] = centers + new_lengths / 2
        elif self.location == "start":
            segments[:, 1] = segments[:, 0] + new_lengths
        elif self.location == 'end':
            segments[:, 0] = segments[:, 1] - new_lengths
        else:
            raise ValueError(f"Unknown location: '{self.location}'")
        return segments

    def __call__(self, segments: np.ndarray) -> np.ndarray:
        """
        Args:
            segments: tensor of shape (M, 2) that contains the start and the end of M segments.

        Returns:
            scaled_segments
        """
        if segments.shape[1] != 2:
            raise ValueError("Segments must have shape (M, 2)")
        scaled_segments = segments.copy()
        lengths = segments[:, 1] - segments[:, 0]
        new_lengths = lengths * self.factor
        new_lengths = np.clip(new_lengths, self.min_length, self.max_length)
        return self._resize_segments(scaled_segments, lengths, new_lengths)


class RandomRelativeScaleSegments(ScaleSegments):
    def __init__(
            self,
            scale_std: float = 0.5,
            location: Literal["center", "start", "end"] = "center",
            min_length: float = 1.0,
            max_length: float = 1000.0,
    ):
        super().__init__(factor=1.0, location=location, min_length=min_length, max_length=max_length)
        self.std = scale_std

    def __call__(self, segments: np.ndarray) -> np.ndarray:
        if segments.shape[1] != 2:
            raise ValueError("Segments must have shape (M, 2)")
        scaled_segments = segments.copy()
        lengths = segments[:, 1] - segments[:, 0]
        new_lengths = (self.std * np.random.randn(lengths.shape[0]) + 1) * lengths
        return self._resize_segments(scaled_segments, lengths, new_lengths)




