import numpy as np

from sign_language_tools.core.transform import Transform


import numpy as np

from sign_language_tools.core.transform import Transform


class RemoveOverlapping(Transform):
    """Remove overlap between segments and enforce a minimum frame gap.

    Segments use inclusive-end convention: [start, end] covers frames
    start, start+1, ..., end, so length = end - start + 1.

    Two segments A, B (sorted by start) are considered too close when
    B.start - A.end <= min_gap. The boundary is placed near the midpoint
    of the overlap/touch region; when min_gap is odd, the extra empty
    frame falls on B's side.
    """

    def __init__(self, min_gap: int = 0):
        super().__init__()
        assert min_gap >= 0, "Minimum temporal gap must be non-negative."
        self.min_gap = min_gap

    def __call__(self, segments: np.ndarray) -> np.ndarray:
        """
        Args:
            segments: Array of shape (N, 2) or (N, K) with K > 2; only the
                first two columns (start, end) are modified, extras pass through.

        Returns:
            Segments with overlaps resolved, sorted by start.
        """
        if len(segments) <= 1:
            return segments

        segments = segments[np.argsort(segments[:, 0])].copy()
        end_prev = segments[:-1, 1]
        start_next = segments[1:, 0]
        overlap_idx = np.where(start_next - end_prev <= self.min_gap)[0]

        if len(overlap_idx) > 0:
            # Midpoint of the contested region between end_prev and start_next.
            midpoints = (end_prev[overlap_idx] + start_next[overlap_idx]) // 2

            # A keeps frames up to (midpoint - left_pad), B starts at
            # (midpoint + right_pad + 1). The +1 ensures at least one
            # frame separates them when min_gap=0 (touching, not overlapping).
            left_pad = self.min_gap // 2
            right_pad = self.min_gap - left_pad

            segments[overlap_idx, 1] = midpoints - left_pad
            segments[overlap_idx + 1, 0] = midpoints + right_pad + 1

            # Length clamp: under inclusive convention, length >= 1 means end >= start.
            segments[:, 1] = np.maximum(segments[:, 0], segments[:, 1])

        return segments


if __name__ == "__main__":
    _segments = np.array([
        [2, 5],
        [6, 14],
        [12, 17],
    ])

    transform = RemoveOverlapping(min_gap=1)
    print(transform(_segments))