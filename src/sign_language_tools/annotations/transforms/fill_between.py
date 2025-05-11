import numpy as np

from sign_language_tools.core.transform import Transform


class FillBetween(Transform):
    def __init__(
        self,
        start_value: int = 1,
        end_value: int = 1,
        fill_value: int = 1,
        max_width: int | None = None,
        bidirectional: bool = False,
    ):
        """Initializes the FillBetween transform.

        Args:
            start_value: The label of the segments where the filling starts. Defaults to 1.
            end_value: The label of the segments where the filling ends. Defaults to 1.
            fill_value: The label to use for the newly created segments
                that fill the gaps. Defaults to 1.
            max_width: The maximum width (duration) of a gap to be filled.
                If None, all identified gaps will be filled. Defaults to None.
            bidirectional: If True, also considers gaps between 'end' and 'start' segments,
                and not only from start to end. Defaults to False.
        """

        super().__init__()
        self.start_value = start_value
        self.end_value = end_value
        self.fill_value = fill_value
        self.max_width = max_width
        self.bidirectional = bidirectional

    def __call__(self, segments: np.ndarray) -> np.ndarray:
        if segments.shape[0] < 2:
            return segments
        segments = segments[segments[:, 0].argsort(axis=0)].copy()
        segment_indices = np.arange(len(segments))
        # Compute all transitions with their corresponding pairs of segments
        segment_pairs = np.column_stack(
            (
                segment_indices[:-1],
                segment_indices[1:],
            )
        )
        transitions = np.column_stack(
            (
                segments[segment_pairs[:, 0], 2],  # Labels from first segments
                segments[segment_pairs[:, 1], 2],  # Labels from second segments
            )
        )
        # Only keep segments with the right transitions
        if self.bidirectional:
            selected_pairs = segment_pairs[
                (
                    (transitions[:, 0] == self.start_value)
                    & (transitions[:, 1] == self.end_value)
                )
                | (
                    (transitions[:, 1] == self.start_value)
                    & (transitions[:, 0] == self.end_value)
                )
            ]
        else:
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
        # Filter out filling segments that are too long
        if self.max_width is not None:
            filling_segments = filling_segments[
                filling_segments[:, 1] - filling_segments[:, 0] <= self.max_width
            ]
        segments = np.concatenate((segments, filling_segments), axis=0)
        segments = segments[segments[:, 0].argsort(axis=0)]
        return segments


if __name__ == "__main__":
    _segments = np.array(
        [
            [2, 5, 1],
            [14, 17, 1],
            [30, 34, 1],
            [40, 52, 1],
        ]
    )

    transform = FillBetween(start_value=1, end_value=1, fill_value=1)
    _segments = transform(_segments)
    print(_segments)
