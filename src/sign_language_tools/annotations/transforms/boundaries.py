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

    Boundary placement uses integer arithmetic when `width` is given as an
    integer: a boundary of width `w` centered on transition `t` covers
    `[t - w // 2, t - w // 2 + w - 1]`. For even widths this is asymmetric
    by one frame (more on the right of the transition than the left), which
    keeps everything on the frame grid without rounding artifacts.

    The emitted boundaries are `[b_start, b_end, label]` triples. The output
    is sorted by `b_start` and may contain overlapping or duplicate
    boundaries when segments are close together — downstream code should
    handle this if needed. Boundaries that fall entirely outside
    `[min_start, max_end]` after clamping are dropped.

    Args:
        width: Fixed boundary width in frames. Exactly one of `width` or
            `relative_width` must be provided.
        relative_width: Boundary width as a fraction of each segment's
            length, rounded to the nearest integer. Exactly one of
            `width` or `relative_width` must be provided.
        min_width: Lower bound on the boundary width, applied after
            `relative_width` is resolved. Defaults to 1 so very short
            segments still produce a visible boundary.
        exclude_start: If True, do not emit start-boundaries.
        exclude_end: If True, do not emit end-boundaries.
        min_start: If not None, boundary starts are clamped to at least
            this value. Typically 0 to keep boundaries inside the
            timeline.
        max_end: If not None, boundary ends are clamped to at most this
            value. Set this to the sequence length minus one to keep
            boundaries inside the timeline.
        boundary_labels: Labels assigned to (start-boundaries,
            end-boundaries) respectively. Using distinct values (e.g.
            `(1, 2)`) lets downstream code tell the two boundary types
            apart.

    Example:
        >>> import numpy as np
        >>> from sign_language_tools.annotations.transforms import SegmentsToBoundaries
        >>> segments = np.array([[0, 6], [7, 9], [13, 16]])  # (M, 2)
        >>> transform = SegmentsToBoundaries(width=4, boundary_labels=(1, 2))
        >>> transform(segments)
        array([[ 0,  1,  1],
               [ 5,  8,  1],
               [ 5,  8,  2],
               [ 8, 11,  2],
               [11, 14,  1],
               [15, 18,  2]])
    """

    def __init__(
        self,
        width: int | None = None,
        relative_width: float | None = None,
        min_width: int = 1,
        exclude_start: bool = False,
        exclude_end: bool = False,
        min_start: int | None = 0,
        max_end: int | None = None,
        boundary_labels: tuple[int, int] = (1, 1),
    ):
        if (width is None) == (relative_width is None):
            raise ValueError(
                "Exactly one of `width` or `relative_width` must be specified."
            )
        if width is not None and width < 1:
            raise ValueError("`width` must be at least 1.")
        if relative_width is not None and relative_width <= 0:
            raise ValueError("`relative_width` must be positive.")
        if min_width < 1:
            raise ValueError("`min_width` must be at least 1.")

        super().__init__()
        self.width = width
        self.relative_width = relative_width
        self.min_width = min_width
        self.exclude_start = exclude_start
        self.exclude_end = exclude_end
        self.min_start = min_start
        self.max_end = max_end
        self.boundary_labels = boundary_labels

    def __call__(self, segments: np.ndarray) -> np.ndarray:
        """Transforms segments into their transition boundaries.

        Args:
            segments: Array of shape `(M, 2)` or `(M, K)` with `K > 2`
                containing at least the start and end of `M` segments;
                extra columns are ignored.

        Returns:
            Array of shape `(M', 3)` where each row is
            `[boundary_start, boundary_end, label]`, sorted by
            `boundary_start`. The dtype matches the input dtype.
        """
        # Always return a (0, 3) array for empty input: the output schema
        # is [start, end, label] regardless of the input's column count.
        if segments.shape[0] == 0:
            return np.zeros((0, 3), dtype=segments.dtype)

        starts = segments[:, 0]
        ends = segments[:, 1]
        lengths = ends - starts + 1
        start_label, end_label = self.boundary_labels

        # Collect transitions with their per-transition metadata. End-
        # transitions sit on `ends + 1` so a boundary centered there marks
        # the step *out of* the segment, symmetric with start-transitions.
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
        source_lengths = np.concatenate(length_parts)
        labels = np.concatenate(label_parts)

        # Resolve boundary widths as integers on the frame grid.
        if self.width is not None:
            boundary_widths = np.full(
                transitions.shape, self.width, dtype=np.int64
            )
        else:
            boundary_widths = np.round(
                self.relative_width * source_lengths
            ).astype(np.int64)
        boundary_widths = np.maximum(boundary_widths, self.min_width)

        # Center the boundary on the transition. For width w, the boundary
        # covers [t - w//2, t - w//2 + w - 1]. For odd w this is symmetric;
        # for even w it extends one frame further to the right of t than
        # to the left.
        left = boundary_widths // 2
        b_starts = transitions - left
        b_ends = b_starts + boundary_widths - 1

        boundaries = np.stack([b_starts, b_ends, labels], axis=1)

        # Clamp to [min_start, max_end] without producing inverted intervals.
        # A boundary fully outside the valid range is dropped rather than
        # collapsed to a degenerate point.
        if self.min_start is not None:
            valid = boundaries[:, 1] >= self.min_start
            boundaries = boundaries[valid]
            boundaries[:, 0] = np.maximum(boundaries[:, 0], self.min_start)
        if self.max_end is not None:
            valid = boundaries[:, 0] <= self.max_end
            boundaries = boundaries[valid]
            boundaries[:, 1] = np.minimum(boundaries[:, 1], self.max_end)

        boundaries = boundaries[np.argsort(boundaries[:, 0])]
        return boundaries.astype(segments.dtype)


if __name__ == "__main__":
    print("=== Default params, distinct labels ===")
    transform = SegmentsToBoundaries(width=4, boundary_labels=(1, 2))
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
        SegmentsToBoundaries(
            relative_width=0.5, min_width=2, boundary_labels=(1, 2)
        )(_segments)
    )

    print("\n=== single-frame segment ===")
    print(transform(np.array([[5, 5]])))

    print("\n=== max_end clamping drops out-of-range boundaries ===")
    print(SegmentsToBoundaries(width=4, max_end=10)(_segments))