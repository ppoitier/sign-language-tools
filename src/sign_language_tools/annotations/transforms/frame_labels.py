import numpy as np

from sign_language_tools.core.transform import Transform


class SegmentsToFrameLabels(Transform):
    """Renders segments as a dense per-frame label vector.

    Segments use inclusive-end convention: `[start, end]` covers frames
    `start, ..., end`. For `(M, 2)` segments, all frames within a segment
    get `fill_label`. For `(M, 3)` segments, the third column is used as
    the per-segment label. Frames not covered by any segment get
    `background_label`. When segments overlap, later segments (in array
    order) overwrite earlier ones for the overlapping frames.

    Args:
        vector_size: Length of the output label vector. If None, inferred
            from the segments as `max(end) + 1`. Pass explicitly when
            frames may extend past the last annotated segment.
        background_label: Label assigned to frames outside any segment.
        fill_label: Label assigned to frames inside a segment when
            `segments` has no label column (shape `(M, 2)`).
        dtype: Output dtype of the label vector.

    Example:
        >>> import numpy as np
        >>> from sign_language_tools.annotations.transforms import SegmentsToFrameLabels
        >>> segments = np.array([[2, 4, 1], [6, 8, 2]])  # (M, 3)
        >>> transform = SegmentsToFrameLabels()
        >>> transform(segments)
        array([0, 0, 1, 1, 1, 0, 2, 2, 2], dtype=int64)
    """

    def __init__(
        self,
        vector_size: int | None = None,
        background_label: int = 0,
        fill_label: int = 1,
        dtype: np.dtype = np.int64,
    ):
        super().__init__()
        self.vector_size = vector_size
        self.background_label = background_label
        self.fill_label = fill_label
        self.dtype = dtype

    def __call__(
        self, segments: np.ndarray, vector_size: int | None = None
    ) -> np.ndarray:
        """Renders the segments as a label vector.

        Args:
            segments: Array of shape `(M, 2)` or `(M, 3)` containing the
                start, end, and optionally the label of `M` segments.
            vector_size: Overrides the `vector_size` passed to the
                constructor for this call only.

        Returns:
            Label vector of shape `(vector_size,)`.
        """
        size = vector_size if vector_size is not None else self.vector_size
        if size is None:
            size = int(segments[:, 1].max()) + 1 if len(segments) else 0

        labels = np.full(size, self.background_label, dtype=self.dtype)
        if len(segments) == 0:
            return labels

        has_labels = segments.shape[1] >= 3
        starts = np.clip(segments[:, 0].astype(int), 0, size)
        ends = np.clip(segments[:, 1].astype(int), 0, size)
        values = segments[:, 2].astype(self.dtype) if has_labels else None

        for i in range(len(segments)):
            v = values[i] if has_labels else self.fill_label
            labels[starts[i] : ends[i] + 1] = v

        return labels


class FrameLabelsToSegments(Transform):
    """Inverse of `SegmentsToFrameLabels`: groups a label vector into segments.

    Contiguous runs of identical labels become `[start, end, label]`
    segments, using the same inclusive-end convention as the rest of the
    module (`[start, end]` covers frames `start, ..., end`). Runs whose
    label is in `background_classes` are excluded from the output.

    Args:
        background_classes: Labels considered background; segments with
            one of these labels are dropped from the output.
        include_labels: If True, the output segments include the label as
            a third column. If False, only `[start, end]` is returned.

    Example:
        >>> import numpy as np
        >>> from sign_language_tools.annotations.transforms import FrameLabelsToSegments
        >>> labels = np.array([0, 0, 1, 1, 1, 0, 2, 2, 2])
        >>> transform = FrameLabelsToSegments()
        >>> transform(labels)
        array([[2, 4, 1],
               [6, 8, 2]])
    """

    def __init__(
        self,
        background_classes: tuple[int, ...] = (0,),
        include_labels: bool = True,
    ):
        super().__init__()
        self.background_classes = background_classes
        self.include_labels = include_labels

    def __call__(self, labels: np.ndarray) -> np.ndarray:
        """Groups the label vector into segments.

        Args:
            labels: Label vector of shape `(T,)`, where `T` is the number
                of frames.

        Returns:
            Array of shape `(M, 3)`, or `(M, 2)` if `include_labels` is
            False, containing the `M` non-background segments.
        """
        if len(labels) == 0:
            cols = 3 if self.include_labels else 2
            return np.empty((0, cols), dtype=np.int64)

        change_idx = np.nonzero(np.diff(labels))[0] + 1
        change_idx = np.r_[0, change_idx, len(labels)]
        starts = change_idx[:-1]
        ends = change_idx[1:] - 1
        values = labels[starts]

        segments = np.stack([starts, ends, values], axis=1)
        keep = ~np.isin(values, self.background_classes)
        segments = segments[keep]

        return segments if self.include_labels else segments[:, :2]
