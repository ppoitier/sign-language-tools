import numpy as np

from sign_language_tools.core.transform import Transform


class MergeSegmentsOnTransition(Transform):
    """Merges adjacent segment pairs whose labels match a given transition.

    Segments are sorted by start. For each pair of adjacent segments (the
    end of one immediately followed by the start of the next, with no
    gap), if their `(label_a, label_b)` pair is listed in `transitions`,
    the pair is merged into a single segment spanning both, labeled
    `new_value`.

    Note:
        This is a single-pass, pairwise transform: a chain of 3 or more
        consecutive segments that all match a transition is merged as
        overlapping pairs rather than collapsed into one segment. In
        practice this only matters when the same transition can repeat
        back-to-back.

    Args:
        transitions: List of `(label_a, label_b)` pairs. A pair of
            adjacent segments is merged when their labels match one of
            these pairs, in order.
        new_value: Label assigned to the merged segments.

    Example:
        >>> import numpy as np
        >>> from sign_language_tools.annotations.transforms import MergeSegmentsOnTransition
        >>> segments = np.array([[2, 5, 1], [6, 10, 2], [12, 15, 3]])  # (M, 3)
        >>> transform = MergeSegmentsOnTransition(transitions=[(1, 2)], new_value=9)
        >>> transform(segments)
        array([[ 2, 10,  9],
               [12, 15,  3]])
    """

    def __init__(self, transitions: list[tuple[int, int]], new_value: int = 1):
        super().__init__()
        self.transitions = np.array(transitions)
        self.new_value = new_value

    def __call__(self, segments: np.ndarray) -> np.ndarray:
        """Merges matching adjacent segment pairs.

        Args:
            segments: Array of shape `(M, 3)` containing the start, end,
                and label of `M` segments.

        Returns:
            The segments after merging, sorted by start.
        """
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


class MergeSegments(Transform):
    """Merges overlapping or touching segments into single segments.

    Segments are sorted by start, then greedily merged: a segment is
    folded into the current run whenever its start falls at or before
    `current_end + 1` (i.e. it overlaps or touches the run with no gap).
    If `segments` has a label column, the merged segment keeps the label
    of the first segment in each run.

    Example:
        >>> import numpy as np
        >>> from sign_language_tools.annotations.transforms import MergeSegments
        >>> segments = np.array([
        ...     [2, 5, 1],
        ...     [6, 12, 2],
        ...     [14, 17, 0],
        ...     [18, 25, 1],
        ...     [67, 78, 3],
        ... ])  # (M, 3)
        >>> transform = MergeSegments()
        >>> transform(segments)
        array([[ 2, 12,  1],
               [14, 25,  0],
               [67, 78,  3]])
    """

    def __init__(self):
        super().__init__()

    def __call__(self, segments: np.ndarray) -> np.ndarray:
        """Merges the overlapping or touching segments.

        Args:
            segments: Array of shape `(M, 2)` or `(M, 3)` containing the
                start and end (and optionally a label) of `M` segments.

        Returns:
            The merged segments, sorted by start.
        """
        if segments.shape[0] < 2:
            return segments
        segments = segments[segments[:, 0].argsort(axis=0)].copy()
        has_labels = segments.shape[1] >= 3

        merged_rows = []
        current_start = segments[0, 0]
        current_end = segments[0, 1]
        current_label = segments[0, 2] if has_labels else None

        for i in range(1, segments.shape[0]):
            next_start = segments[i, 0]
            next_end = segments[i, 1]
            if next_start <= current_end + 1:
                current_end = max(current_end, next_end)
            else:
                row = [current_start, current_end, current_label] if has_labels else [current_start, current_end]
                merged_rows.append(row)
                current_start = next_start
                current_end = next_end
                if has_labels:
                    current_label = segments[i, 2]

        row = [current_start, current_end, current_label] if has_labels else [current_start, current_end]
        merged_rows.append(row)
        return np.stack(merged_rows)


if __name__ == "__main__":
    _segments = np.array([
        [2, 5, 1],
        [6, 12, 2],
        [14, 17, 0],
        [18, 25, 1],
        [67, 78, 3],
    ])

    transform = MergeSegments()
    _segments = transform(_segments)
    print(_segments)
