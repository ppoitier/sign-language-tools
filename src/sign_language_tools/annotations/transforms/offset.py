from typing import Literal

import numpy as np

from sign_language_tools.core.transform import Transform


class LinearBoundaryOffset(Transform):
    def __init__(
        self,
        sequence_length: int,
        ref_location: Literal["start", "end"] = "start",
        background_class: int = -1,
    ):
        """
        Transform segments into a time series where each time t that is contained into a segment is
        the offset (distance) between and a reference location (start or end of the segment).

        Args:
            sequence_length: The length of the resulting time series. It should ideally be greater that the last boundary.
            ref_location: The reference of the offset when in a segment. Can be 'start' or 'end'. Default is start.
            background_class: The value at each t when not in any segment. Default is -1.
        """
        super().__init__()
        self.sequence_length = sequence_length
        self.ref_location = ref_location
        self.background_class = background_class

    def __call__(self, segments: np.ndarray) -> np.ndarray:
        """
        Args:
            segments: array of shape (M, 2) for the start and end of M segments.

        Returns:
            time_series_with_offsets
        """
        time_series = np.full(
            self.sequence_length, self.background_class, dtype=np.float32
        )
        if len(segments) == 0:
            return time_series
        segments = segments[segments[:, 0].argsort()].copy()
        time_indices = np.arange(self.sequence_length)
        for start, end in segments[:, :2]:
            start_idx = int(start)
            end_idx = int(end)
            if start_idx >= self.sequence_length:
                continue
            end_idx = min(end_idx, self.sequence_length)
            if self.ref_location == "start":
                time_series[start_idx:end_idx] = (
                    time_indices[start_idx:end_idx] - start_idx
                )
            elif self.ref_location == "end":
                time_series[start_idx:end_idx] = (
                    end_idx - time_indices[start_idx:end_idx]
                )
            else:
                raise ValueError(f"Unknown reference location: {self.ref_location}")
        return time_series


if __name__ == "__main__":
    transform = LinearBoundaryOffset(
        sequence_length=20, ref_location="start", background_class=-1
    )
    segments = np.array(
        [
            [5, 10],
            [15, 18],
        ]
    )
    result = transform(segments)
    print("Result:", result)
