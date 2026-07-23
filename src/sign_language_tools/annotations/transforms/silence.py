import numpy as np

from sign_language_tools.core.transform import Transform


class CloseShortSilences(Transform):
    """Close short silences between consecutive segments by extending them.

    Segments use inclusive-end convention: [start, end] covers frames
    start, ..., end. The silence between two segments [a, b] and [c, d]
    (with c > b) is `c - b - 1` empty frames.

    For each adjacent pair whose silence is at most `max_silence` frames,
    the boundary is placed near the midpoint of the gap. When the silence
    has odd length, the extra frame is absorbed into the earlier segment.

    Like `RemoveOverlapping`, this is a single-pass transform: cascading
    effects (where closing one gap might shift a boundary into another
    nearby gap) are not handled. In practice this only matters when
    `max_silence` is large relative to segment lengths.

    Args:
        max_silence: Maximum number of empty frames between two segments
            for the gap to be closed. Must be non-negative.

    Example:
        >>> import numpy as np
        >>> from sign_language_tools.annotations.transforms import CloseShortSilences
        >>> segments = np.array([[0, 6], [8, 13], [25, 27], [30, 41]])  # (M, 2)
        >>> transform = CloseShortSilences(max_silence=3)
        >>> transform(segments)
        array([[ 0,  7],
               [ 8, 13],
               [25, 28],
               [29, 41]])
    """

    def __init__(self, max_silence: int):
        super().__init__()
        assert max_silence >= 0, "max_silence must be non-negative."
        self.max_silence = max_silence

    def __call__(self, segments: np.ndarray) -> np.ndarray:
        """Closes short silences between consecutive segments.

        Args:
            segments: Array of shape `(M, 2)` containing the start and end
                of `M` segments.

        Returns:
            The segments with short silences closed, sorted by start.
        """
        if segments.shape[0] < 2:
            return segments.copy()

        segments = segments[np.argsort(segments[:, 0])]
        end_prev = segments[:-1, 1]
        start_next = segments[1:, 0]
        silences = start_next - end_prev - 1
        close_idx = np.where((silences >= 0) & (silences <= self.max_silence))[0]

        if len(close_idx) > 0:
            # Midpoint of the silent frames. For a silence covering frames
            # [end_prev + 1, start_next - 1], the midpoint frame is
            # (end_prev + start_next) // 2. The earlier segment extends up
            # to and including this frame; the later segment starts on the
            # next frame.
            midpoints = (end_prev[close_idx] + start_next[close_idx]) // 2
            segments[close_idx, 1] = midpoints
            segments[close_idx + 1, 0] = midpoints + 1

        return segments


if __name__ == "__main__":
    transform = CloseShortSilences(max_silence=3)
    segments = np.array(
        [
            [0, 6],
            [8, 13],
            [30, 41],
            [25, 27],
        ]
    )
    segments = transform(segments)
    print(segments)
