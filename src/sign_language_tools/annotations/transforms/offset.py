import numpy as np

from sign_language_tools.core.transform import Transform


class SegmentsToBoundaryOffsets(Transform):
    """Render segments as a per-frame `(start_offset, end_offset)` series.

    Segments use inclusive-end convention: [start, end] covers frames
    start, ..., end. For each frame t inside a segment:
      * `start_offset[t] = t - start`   (frames since segment began)
      * `end_offset[t]   = end - t`     (frames until segment ends)
    Both offsets are 0 at the boundary frames and positive in between.
    Frames not covered by any segment get `background_value`.

    When segments overlap, later segments overwrite earlier ones for the
    overlapping frames (sorted by start). To avoid surprises, run
    `RemoveOverlapping` upstream.

    Parameters
    ----------
    sequence_length : int, optional
        Length of the output series. If None, inferred from the segments
        as `max(end) + 1`. Pass explicitly when frames may extend past
        the last annotated segment.
    background_value : float, default -1.0
        Value for frames outside any segment, written to both columns.
    dtype : np.dtype, default np.float32
        Output dtype. Float because offsets are commonly used as
        regression targets, and the background sentinel may be non-integer.

    Returns
    -------
    np.ndarray of shape (sequence_length, 2)
    """

    def __init__(
        self,
        sequence_length: int | None = None,
        background_value: float = -1.0,
        dtype: np.dtype = np.float32,
    ):
        super().__init__()
        self.sequence_length = sequence_length
        self.background_value = background_value
        self.dtype = dtype

    def __call__(
        self, segments: np.ndarray, sequence_length: int | None = None
    ) -> np.ndarray:
        size = (
            sequence_length
            if sequence_length is not None
            else self.sequence_length
        )
        if size is None:
            size = int(segments[:, 1].max()) + 1 if len(segments) else 0

        offsets = np.full((size, 2), self.background_value, dtype=self.dtype)
        if len(segments) == 0:
            return offsets

        time_indices = np.arange(size)
        segments = segments[np.argsort(segments[:, 0])]

        for start, end in segments[:, :2].astype(int):
            s = max(0, start)
            e = min(size - 1, end)  # inclusive
            if s > e:
                continue
            offsets[s:e + 1, 0] = time_indices[s:e + 1] - start
            offsets[s:e + 1, 1] = end - time_indices[s:e + 1]

        return offsets


if __name__ == "__main__":
    transform = SegmentsToBoundaryOffsets(sequence_length=20)
    segments = np.array(
        [
            [5, 10],
            [15, 18],
        ]
    )
    result = transform(segments)
    print("Result:", result)
