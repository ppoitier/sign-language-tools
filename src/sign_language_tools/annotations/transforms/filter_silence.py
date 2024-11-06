import numpy as np

from sign_language_tools.core.transform import Transform


class FilterShortSilence(Transform):
    def __init__(self, min_duration: float):
        super().__init__()
        self.min_duration = min_duration

    def __call__(self, segments: np.ndarray) -> np.ndarray:
        if segments.shape[0] < 2:
            return segments
        segments = segments[segments[:, 0].argsort(axis=0)].copy()
        gaps = segments[1:, 0] - segments[:-1, 1] - 1

        for index, gap in enumerate(gaps):
            if gap <= self.min_duration:
                segments[index, 1] += gap / 2
                segments[index+1, 0] -= gap / 2

        return segments


if __name__ == "__main__":
    transform = FilterShortSilence(min_duration=3)
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
