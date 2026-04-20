import numpy as np

from sign_language_tools.core.transform import Transform


class SegmentsToBoundaries(Transform):
    """Transform segments into boundary regions centered on their transitions.

    A *segment* is a `[start, end]` interval (both inclusive, in frames) marking
    an event in a sequence — for example, a sign in a sign-language video. A
    *boundary* is a short interval centered on a transition (the point where a
    segment starts or ends) that can be used as a training target for models
    that detect segment onsets/offsets rather than segment content.

    For each input segment, this transform can emit up to two boundaries:
      * a **start-boundary** centered on the segment's first frame, and
      * an **end-boundary** centered on the frame *immediately after* the
        segment's last frame (i.e. on `end + 1`).

    Centering the end-boundary on `end + 1` rather than on `end` means the
    boundary marks the transition *out of* the segment (between the last
    in-segment frame and the first out-of-segment frame), which is symmetric
    with the start-boundary marking the transition *into* the segment.

    The emitted boundaries are `[b_start, b_end, label]` triples where
    `b_end - b_start + 1` equals the boundary width (before clamping). The
    output is **not** sorted and may contain overlapping or duplicate
    boundaries when segments are close together — downstream code should
    handle this if needed.

    Parameters
    ----------
    width : float, optional
        Fixed boundary width in frames. Exactly one of `width` or
        `relative_width` must be provided.
    relative_width : float, optional
        Boundary width as a fraction of each segment's length. Exactly one
        of `width` or `relative_width` must be provided. Note that when the
        output is cast back to an integer dtype, fractional widths are
        truncated.
    min_width : float, default 0.0
        Lower bound on the boundary width, applied after `relative_width` is
        resolved. Useful when very short segments would otherwise produce
        zero-width boundaries.
    exclude_start : bool, default False
        If True, do not emit start-boundaries.
    exclude_end : bool, default False
        If True, do not emit end-boundaries.
    min_start : float or None, default 0.0
        If not None, boundary starts are clamped to at least this value.
        Typically 0.0 to keep boundaries inside the timeline.
    max_end : float or None, default None
        If not None, boundary ends are clamped to at most this value.
        Set this to the sequence length minus one to keep boundaries inside
        the timeline.
    boundary_label : tuple of (int, int), default (1, 1)
        Labels assigned to (start-boundaries, end-boundaries) respectively.
        Using distinct values (e.g. `(1, 2)`) lets downstream code tell the
        two boundary types apart.

    Returns
    -------
    np.ndarray of shape (n_boundaries, 3)
        Each row is `[boundary_start, boundary_end, label]`. The dtype
        matches the input dtype; with an integer input dtype, fractional
        widths are truncated.

    Examples
    --------
    >>> transform = SegmentsToBoundaries(width=4, boundary_label=(1, 2))
    >>> segments = np.array([[0, 6], [7, 9], [13, 16]])
    >>> transform(segments)
    array([[ 0,  1,  1],
           [ 5,  8,  1],
           [11, 14,  1],
           [ 5,  8,  2],
           [ 8, 11,  2],
           [15, 18,  2]])
    """

    def __init__(
        self,
        width: float | None = None,
        relative_width: float | None = None,
        min_width: float = 0.0,
        exclude_start: bool = False,
        exclude_end: bool = False,
        min_start: float | None = 0.0,
        max_end: float | None = None,
        boundary_label: tuple[int, int] = (1, 1),
    ):
        if (width is None) == (relative_width is None):
            raise ValueError(
                "Exactly one of `width` or `relative_width` must be specified."
            )

        super().__init__()
        self.width = width
        self.relative_width = relative_width
        self.min_width = min_width
        self.exclude_start = exclude_start
        self.exclude_end = exclude_end
        self.min_start = min_start
        self.max_end = max_end
        self.boundary_label = boundary_label

    def __call__(self, segments: np.ndarray) -> np.ndarray:
        # Always return a (0, 3) array for empty input: the output schema is
        # [start, end, label] regardless of the input's column count.
        if segments.shape[0] == 0:
            return np.zeros((0, 3), dtype=segments.dtype)

        segments = segments[np.argsort(segments[:, 0])]
        starts = segments[:, 0]
        ends = segments[:, 1]
        lengths = ends - starts + 1
        start_label, end_label = self.boundary_label

        # Build the list of transition points and their per-transition
        # metadata (source segment length, label). End-transitions sit on
        # `ends + 1` so that a boundary centered there marks the step out
        # of the segment, symmetric with start-transitions on `starts`.
        transition_parts = []
        length_parts = []
        label_parts = []

        if not self.exclude_start:
            transition_parts.append(starts)
            length_parts.append(lengths)
            label_parts.append(np.full(len(starts), start_label))

        if not self.exclude_end:
            transition_parts.append(ends + 1)
            length_parts.append(lengths)
            label_parts.append(np.full(len(ends), end_label))

        if not transition_parts:
            return np.zeros((0, 3), dtype=segments.dtype)

        transitions = np.concatenate(transition_parts)
        lengths = np.concatenate(length_parts)
        labels = np.concatenate(label_parts)

        if self.width is not None:
            boundary_lengths = np.full(transitions.shape, self.width, dtype=float)
        else:
            boundary_lengths = self.relative_width * lengths.astype(float)
        boundary_lengths = np.maximum(boundary_lengths, self.min_width)

        # Boundaries are centered on transitions. The `-1` on the end makes
        # the inclusive length equal to `boundary_lengths` (since both
        # endpoints are inclusive in the segment convention).
        trans_starts = transitions - boundary_lengths / 2
        trans_ends = transitions + boundary_lengths / 2 - 1

        boundaries = np.stack((trans_starts, trans_ends, labels), axis=1)
        if self.min_start is not None:
            boundaries[:, 0] = np.maximum(boundaries[:, 0], self.min_start)
        if self.max_end is not None:
            boundaries[:, 1] = np.minimum(boundaries[:, 1], self.max_end)
        boundaries = boundaries[np.argsort(boundaries[:, 0])]
        return boundaries.astype(segments.dtype)


if __name__ == "__main__":
    print("=== Default params, distinct labels ===")
    transform = SegmentsToBoundaries(width=4, boundary_label=(1, 2))
    _segments = np.array(
        [
            [0, 6],
            [7, 9],
            [13, 16],
        ]
    )
    print(transform(_segments))

    print("\n=== exclude_end=True (only starts) ===")
    print(SegmentsToBoundaries(width=4, exclude_end=True)(_segments))

    print("\n=== exclude_start=True (only ends) ===")
    print(SegmentsToBoundaries(width=4, exclude_start=True)(_segments))

    print("\n=== both excluded (empty result) ===")
    print(
        SegmentsToBoundaries(width=4, exclude_start=True, exclude_end=True)(_segments)
    )

    print("\n=== empty input ===")
    print(transform(np.array([]).reshape(0, 2)))

    print("\n=== relative_width=0.5 with min_width=2 ===")
    print(
        SegmentsToBoundaries(relative_width=0.5, min_width=2, boundary_label=(1, 2))(
            _segments
        )
    )
