from typing import Union

import numpy as np


def intersection_over_union(
        interval: Union[tuple[float, float], np.ndarray],
        segments: np.ndarray,
) -> np.ndarray:
    r"""
    Computes the intersection-over-union (IoU) between an interval and K segments.
    The IoU is also called the Jaccard index or the Jaccard similarity coefficient.

    The IoU between an interval A and a segment B is: $\frac{|A \cup B|}{|A \cap B|}$.

    Args:
        interval: An interval which is a tuple (start, end) or a Numpy array of shape (2).
        segments: A Numpy array of shape (K, 2) containing K segments (start, end).

    Returns:
        iou_scores: The IoU score between the interval and each of the K segments.
            This is a Numpy array of shape (K).
    """
    start, end = interval
    seg_start = segments[:, 0]
    seg_end = segments[:, 1]
    seg_len = seg_end - seg_start
    inter_start = np.maximum(seg_start, start)
    inter_end = np.minimum(seg_end, end)
    inter_len = np.maximum(inter_end - inter_start, 0)
    union_len = seg_len - inter_len + end - start
    return inter_len / union_len
