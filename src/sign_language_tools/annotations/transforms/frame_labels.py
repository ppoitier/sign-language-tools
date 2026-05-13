import numpy as np

from sign_language_tools.core.transform import Transform


class SegmentsToFrameLabels(Transform):
    """Render (N, 2) or (N, 3) segments as a per-frame label vector.

    For (N, 2) segments, all frames within a segment get `fill_label`.
    For (N, 3) segments, the third column is used as the per-segment label.
    Frames not covered by any segment get `background_label`.
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
        size = vector_size if vector_size is not None else self.vector_size
        if size is None:
            size = int(segments[:, 1].max()) if len(segments) else 0

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
    """Inverse: contiguous runs of identical labels → (start, end, label) segments.

    Background classes are excluded from the output. End is exclusive.
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
