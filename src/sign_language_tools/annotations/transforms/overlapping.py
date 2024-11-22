from math import ceil, floor

import numpy as np

from sign_language_tools.core.transform import Transform


class RemoveOverlapping(Transform):
    """Remove overlapping between segments by setting their boundaries to the midpoint
    of the overlap, and then add a minimum gap between them.

    When two segments A and B overlap:
    1. Find the midpoint between A[end] and B[start]
    2. Set A[end] to midpoint - floor(min_gap/2)
    3. Set B[start] to midpoint + ceil(min_gap/2)

    This ensures the minimum gap is evenly distributed around the midpoint.
    When min_gap is odd, the extra unit is added to the later segment."""

    def __init__(self, min_gap: int = 1):
        super().__init__()
        assert min_gap > 0, "min_gap must be greater than 0"
        self.min_gap = min_gap

    def __call__(self, segments: np.ndarray) -> np.ndarray:
        """
        Args:
            segments: Array of shape (N, 2) that contains N segments (start, end).

        Returns:
            new_segments: Initial segments, but without overlapping between them.
        """
        if len(segments) <= 1:
            return segments

        # Sort segments by start time
        sorted_idx = np.argsort(segments[:, 0])
        segments = segments[sorted_idx].copy()

        end_prev = segments[:-1, 1]
        start_next = segments[1:, 0]
        overlaps = end_prev + (self.min_gap - 1) >= start_next
        if np.any(overlaps):
            midpoints = (end_prev[overlaps] + start_next[overlaps]) / 2
            segments[:-1][overlaps, 1] = midpoints - floor(self.min_gap / 2)
            segments[1:][overlaps, 0] = midpoints + ceil(self.min_gap / 2)

        return segments


if __name__ == "__main__":
    segments = np.array([
        [2, 5],
        [6, 14],
        [12, 17],
        [41, 44],
        [45, 48],

    ])

    transform = RemoveOverlapping(min_gap=2)
    print(transform(segments))
