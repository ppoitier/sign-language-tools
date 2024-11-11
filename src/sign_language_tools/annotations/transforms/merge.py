import numpy as np

from sign_language_tools.core.transform import Transform


class MergeSegmentsOnTransition(Transform):
    def __init__(self, transitions: list[tuple[int, int]], new_value: int = 1):
        """
        Merge segments when specific transitions occurs.

        Args:
            transitions: list that contains the transitions where the segments are merged.
        """
        super().__init__()
        self.transitions = np.array(transitions)
        self.new_value = new_value

    def __call__(self, segments: np.ndarray) -> np.ndarray:
        if segments.shape[0] < 2:
            return segments
        segments = segments[segments[:, 0].argsort(axis=0)].copy()
        segment_indices = np.arange(len(segments))
        segment_pairs = np.column_stack((
            segment_indices[:-1],  # All indices except last
            segment_indices[1:]  # All indices except first
        ))
        transitions = np.column_stack((
            segments[segment_pairs[:, 0], 2],  # Values from first segments
            segments[segment_pairs[:, 1], 2]  # Values from second segments
        ))
        are_selected = (transitions[:, None] == self.transitions).all(axis=2).any(axis=1)
        are_adjacent = (segments[segment_pairs[:, 0], 1] + 1) == segments[segment_pairs[:, 1], 0]

        to_merge = are_selected & are_adjacent

        # Create merged segments from pairs
        merged = np.column_stack((
            segments[segment_pairs[to_merge, 0], 0],  # Start times from first segments
            segments[segment_pairs[to_merge, 1], 1],  # End times from second segments
            np.full(to_merge.sum(), self.new_value)  # Values from first segments
        ))

        # Get indices of segments that weren't merged
        merged_segment_indices = np.unique(segment_pairs[to_merge].ravel())
        unmerged_mask = ~np.isin(np.arange(len(segments)), merged_segment_indices)
        unmerged = segments[unmerged_mask]

        new_segments = np.vstack((merged, unmerged))
        return new_segments[new_segments[:, 0].argsort(axis=0)]


if __name__ == "__main__":
    segments = np.array([
        [2, 5, 1],
        [6, 12, 2],
        [14, 17, 0],
        [18, 25, 1],
        [67, 78, 3],
    ])

    transform = MergeSegmentsOnTransition([(1, 2)], new_value=45)
    segments = transform(segments)
    print(segments)
