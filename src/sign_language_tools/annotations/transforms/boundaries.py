import numpy as np

from sign_language_tools.core.transform import Transform


class SegmentsToBoundaries(Transform):
    def __init__(
        self,
        width: float | None = None,
        relative_width: float | None = None,
        min_width: float | None = None,
        exclude_start: bool = False,
        exclude_end: bool = False,
        exclude_edges: bool = True,
        min_start: float | None = 0.0,
        max_end: float | None = None,
        rounded: bool = True,
        discrete_gap: bool = True,
    ):
        """
        Transform an array of shape (M, 2) for the start and the end of M segments, to an array of shape (N, 2)
        for the start and the end of N boundaries positioned at the transitions between the segments.

        If there is a gap between two segments, we consider this as a `background` segment.

        Args:
            width: Targeted with of a boundary (centered on the transition between segments).
        """
        assert (width is not None) or (
            relative_width is not None
        ), "You have to specify either width or relative width."

        super().__init__()
        self.width = width
        self.relative_width = relative_width
        self.min_width = min_width
        self.exclude_start = exclude_start
        self.exclude_end = exclude_end
        self.exclude_edges = exclude_edges
        self.min_start = min_start
        self.max_end = max_end
        self.rounded = rounded
        self.discrete_gap = discrete_gap

    def __call__(self, segments: np.ndarray) -> np.ndarray:
        if len(segments) == 0:
            return np.zeros((0, 2))
        # Sort segments by start time
        sorted_idx = np.argsort(segments[:, 0])
        segments = segments[sorted_idx]

        # Calculate segment lengths
        segment_lengths = segments[:, 1] - segments[:, 0] + 1

        # Create arrays for starts and ends with associated lengths
        starts = segments[:, 0]
        ends = segments[:, 1]

        if self.exclude_start:
            starts = np.array([])
        if self.exclude_end:
            ends = np.array([])

        # Stack all transitions and their associated lengths
        if self.discrete_gap:
            transitions = np.concatenate([starts - 0.5, ends + 0.5])
        else:
            transitions = np.concatenate([starts, ends])
        associated_lengths = np.concatenate([segment_lengths, segment_lengths])

        # Get sorting indices for transitions to handle duplicates
        sort_idx = np.argsort(transitions)
        transitions = transitions[sort_idx]
        associated_lengths = associated_lengths[sort_idx]

        # Find unique transitions while preserving the first occurrence
        unique_mask = np.concatenate(
            [np.array([True]), transitions[1:] != transitions[:-1]]
        )
        transitions = transitions[unique_mask]
        associated_lengths = associated_lengths[unique_mask]
        if self.width is not None:
            width_values = np.full_like(transitions, self.width)
        else:
            width_values = self.relative_width * associated_lengths
        if self.min_width is not None:
            width_values[width_values < self.min_width] = self.min_width
        boundaries_start = transitions - width_values / 2
        boundaries_end = transitions + width_values / 2
        boundaries = np.stack((boundaries_start, boundaries_end), axis=1)
        if self.exclude_edges:
            mask = np.ones(len(boundaries), dtype=bool)
            if self.min_start is not None:
                mask &= boundaries[:, 0] > self.min_start
            if self.max_end is not None:
                mask &= boundaries[:, 1] < self.max_end
            boundaries = boundaries[mask]
        if self.rounded:
            return np.round(boundaries).astype('int32')
        return boundaries


if __name__ == "__main__":
    transform = SegmentsToBoundaries(relative_width=0.2)
    segments = np.array(
        [
            [0, 6],
            [7, 9],
            # [30, 41],
            # [25, 29],
        ]
    )
    boundaries = transform(segments)
    print(boundaries)
