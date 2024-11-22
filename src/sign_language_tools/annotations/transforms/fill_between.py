import numpy as np

from sign_language_tools.core.transform import Transform


class FillBetween(Transform):
    def __init__(self, start_value: int = 1, end_value: int = 1, fill_value: int = 1, alternate: bool = True):
        super().__init__()
        self.start_value = start_value
        self.end_value = end_value
        self.fill_value = fill_value
        self.alternate = alternate

    def __call__(self, segments: np.ndarray) -> np.ndarray:
        if segments.shape[0] < 2:
            return segments
        segments = segments[segments[:, 0].argsort(axis=0)].copy()
        segment_indices = np.arange(len(segments))
        # Compute all transitions with their corresponding pairs of segments
        segment_pairs = np.column_stack(
            (
                segment_indices[:-1],  # All indices except last
                segment_indices[1:],  # All indices except first
            )
        )
        if self.alternate:
            segment_pairs = segment_pairs[::2]
        transitions = np.column_stack(
            (
                segments[segment_pairs[:, 0], 2],  # Values from first segments
                segments[segment_pairs[:, 1], 2],  # Values from second segments
            )
        )
        # Only keep segments with the right transitions
        selected_pairs = segment_pairs[
            (transitions[:, 0] == self.start_value)
            & (transitions[:, 1] == self.end_value)
        ]
        # Compute the filling segments
        filling_segments = np.stack(
            (
                segments[selected_pairs[:, 0], 1] + 1,
                segments[selected_pairs[:, 1], 0] - 1,
                np.full((selected_pairs.shape[0],), fill_value=self.fill_value),
            ),
            axis=1,
        )
        segments = np.concatenate((segments, filling_segments), axis=0)
        segments = segments[segments[:, 0].argsort(axis=0)]
        return segments


if __name__ == "__main__":
    segments = np.array(
        [
            [2, 5, 1],
            [14, 17, 1],
            [30, 34, 1],
            [40, 52, 1],
        ]
    )

    transform = FillBetween(start_value=1, end_value=1, fill_value=1)
    segments = transform(segments)
    print(segments)
