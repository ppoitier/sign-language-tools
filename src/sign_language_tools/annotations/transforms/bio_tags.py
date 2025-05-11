import numpy as np
from sign_language_tools.core.transform import Transform


class BioTags(Transform):
    def __init__(self, b_tag_size: int | float = 0.25):
        super().__init__()
        assert b_tag_size > 0, "B tags must be at least 1 unit long."
        self.b_tag_size = b_tag_size

    def __call__(self, segments: np.ndarray) -> np.ndarray:
        n_segments = segments.shape[0]
        starts = segments[:, 0]
        ends = segments[:, 1]
        if isinstance(self.b_tag_size, float):
            lengths = ends - starts + 1
            b_tag_sizes = np.maximum(self.b_tag_size * lengths, 1).round().astype('int32')
        else:
            b_tag_sizes = np.full(n_segments, fill_value=self.b_tag_size)
        b_ends = np.minimum(starts + b_tag_sizes - 1, ends)
        b_labels = np.ones(n_segments, dtype=segments.dtype)
        i_starts = np.minimum(b_ends + 1, ends)
        i_labels = 2 * b_labels
        b_tags = np.stack([starts, b_ends, b_labels], axis=1)
        i_tags = np.stack([i_starts, ends, i_labels], axis=1)
        final_segments = np.concatenate([b_tags, i_tags], axis=0)
        return final_segments[np.argsort(final_segments[:, 0])]


if __name__ == "__main__":
    transform = BioTags(b_tag_size=1)
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
