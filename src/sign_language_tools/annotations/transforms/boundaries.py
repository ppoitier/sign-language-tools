import numpy as np

from sign_language_tools.core.transform import Transform


class SegmentsToBoundaries(Transform):
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
        assert (width is not None) or (
            relative_width is not None
        ), "You have to specify either width or relative width."

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
        if segments.shape[0] == 0:
            return np.zeros((0, segments.shape[1]))
        segments = segments[np.argsort(segments[:, 0])]
        starts = segments[:, 0]
        ends = segments[:, 1]
        lengths = ends - starts + 1
        start_label, end_label = self.boundary_label

        if not self.exclude_start and not self.exclude_end:
            transitions = np.concatenate([starts, ends + 1])
            lengths = np.concatenate([lengths, lengths])
            labels = np.concatenate([
                np.full((len(starts),), fill_value=start_label),
                np.full((len(ends),), fill_value=end_label),
            ])
        elif not self.exclude_start:
            transitions = starts
            labels = np.full((len(starts),), fill_value=start_label)
        elif not self.exclude_end:
            transitions = ends
            labels = np.full((len(ends),), fill_value=end_label)
        else:
            transitions = []

        if self.width is not None:
            boundary_lengths = np.full(transitions.shape, fill_value=self.width)
        elif self.relative_width is not None:
            boundary_lengths = self.relative_width * lengths
        else:
            raise ValueError("Relative width is not supported yet.")
        boundary_lengths = np.maximum(boundary_lengths, self.min_width)

        trans_starts = transitions - boundary_lengths / 2
        trans_ends = transitions + boundary_lengths / 2 - 1

        boundaries = np.stack((trans_starts, trans_ends, labels), axis=1)
        if self.min_start is not None:
            boundaries[:, 0] = np.maximum(boundaries[:, 0], self.min_start)
        if self.max_end is not None:
            boundaries[:, 1] = np.minimum(boundaries[:, 1], self.max_end)
        return boundaries.astype(segments.dtype)


if __name__ == "__main__":
    transform = SegmentsToBoundaries(width=4)
    _segments = np.array(
        [
            [0, 6],
            [7, 9],
            [13, 16],
        ]
    )
    _boundaries = transform(_segments)
    print(_boundaries)
