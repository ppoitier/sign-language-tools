import numpy as np

_KNOWN_UNITS = {"s", "ms", "frame"}


def segments_to_seconds(
    segments: np.ndarray,
    unit: str = "s",
    fps: float = 25.0,
) -> np.ndarray:
    """Convert a (N, 2+) segment array to seconds (float32).

    Only the first two columns (start, end) are kept.
    """
    segs = segments[:, :2].astype("float32")
    if unit == "s":
        return segs
    if unit == "ms":
        return segs / 1000.0
    if unit == "frame":
        return segs / fps
    raise ValueError(f"Unknown unit: '{unit}'. Expected one of {_KNOWN_UNITS}.")