import numpy as np
from sign_language_tools.core.transform import Transform


class BioTags(Transform):
    """Split each segment into a Beginning tag followed by an Inside tag.

    Segments use inclusive-end convention: [start, end] covers frames
    start, ..., end. Each segment of length L produces:
      * a B-tag covering [start, start + w - 1] with label `b_label`, and
      * an I-tag covering [start + w, end] with label `i_label`,
    where w is the B-tag width (either `width` or `round(relative_width * L)`,
    minimum 1). If w >= L the I-tag is omitted, so single-frame segments
    produce a B-tag only.

    Args:
        width: Fixed B-tag width, in frames. Exactly one of `width` or
            `relative_width` must be specified.
        relative_width: B-tag width as a fraction of each segment's
            length, rounded to the nearest integer (minimum 1). Exactly
            one of `width` or `relative_width` must be specified.
        b_label: Label assigned to the B-tag (beginning) of each segment.
        i_label: Label assigned to the I-tag (inside) of each segment.

    Example:
        >>> import numpy as np
        >>> from sign_language_tools.annotations.transforms import BioTags
        >>> segments = np.array([[0, 6], [8, 13]])  # (M, 2)
        >>> transform = BioTags(width=2)
        >>> transform(segments)
        array([[ 0,  1,  1],
               [ 2,  6,  2],
               [ 8,  9,  1],
               [10, 13,  2]])
    """

    def __init__(
        self,
        width: int | None = None,
        relative_width: float | None = None,
        b_label: int = 1,
        i_label: int = 2,
    ):
        if (width is None) == (relative_width is None):
            raise ValueError(
                "Exactly one of `width` or `relative_width` must be specified."
            )
        if width is not None and width < 1:
            raise ValueError("`width` must be at least 1.")
        if relative_width is not None and relative_width <= 0:
            raise ValueError("`relative_width` must be positive.")

        super().__init__()
        self.width = width
        self.relative_width = relative_width
        self.b_label = b_label
        self.i_label = i_label

    def __call__(self, segments: np.ndarray) -> np.ndarray:
        """Splits each segment into B/I tags.

        Args:
            segments: Array of shape `(M, 2)` containing the start and end
                of `M` segments.

        Returns:
            Array of shape `(M', 3)` containing the start, end, and label
            of the resulting B/I tags, sorted by start.
        """
        if segments.shape[0] == 0:
            return np.zeros((0, 3), dtype=segments.dtype)

        starts = segments[:, 0]
        ends = segments[:, 1]
        lengths = ends - starts + 1

        if self.width is not None:
            b_widths = np.full(len(segments), self.width, dtype=np.int64)
        else:
            b_widths = np.maximum(
                np.round(self.relative_width * lengths), 1
            ).astype(np.int64)

        b_ends = np.minimum(starts + b_widths - 1, ends)
        b_tags = np.stack(
            [starts, b_ends, np.full(len(segments), self.b_label)], axis=1
        )

        has_i = b_ends < ends
        i_tags = np.stack(
            [
                b_ends[has_i] + 1,
                ends[has_i],
                np.full(has_i.sum(), self.i_label),
            ],
            axis=1,
        )

        out = np.concatenate([b_tags, i_tags], axis=0)
        out = out[np.argsort(out[:, 0])]
        return out.astype(segments.dtype)


if __name__ == "__main__":
    transform = BioTags(width=2)
    _segments = np.array(
        [
            [0, 6],
            [8, 13],
            [30, 41],
            [25, 27],
        ],
        dtype=np.int32
    )
    bio_tags = transform(_segments)
    print(bio_tags)
