from typing import Union

import numpy as np


def temporal_intersection_over_union(
        target_segment: Union[tuple[float, float], np.ndarray],
        segments: np.ndarray,
) -> np.ndarray:
    r"""
    Computes the temporal intersection-over-union (tIoU) between a target segment and K segments.
    The IoU is also called the Jaccard index or the Jaccard similarity coefficient.

    The tIoU between a segment A and a segment B is: $\frac{|A \cup B|}{|A \cap B|}$.

    Args:
        target_segment: A target segment which is a tuple (start, end) or a Numpy array of shape (2).
        segments: A Numpy array of shape (K, 2) containing K segments (start, end).

    Returns:
        iou_scores: The IoU score between the interval and each of the K segments.
            This is a Numpy array of shape (K).
    """
    start, end = target_segment
    seg_start = segments[:, 0]
    seg_end = segments[:, 1]
    seg_len = seg_end - seg_start
    inter_start = np.maximum(seg_start, start)
    inter_end = np.minimum(seg_end, end)
    inter_len = np.maximum(inter_end - inter_start, 0)
    union_len = seg_len - inter_len + end - start
    return inter_len / union_len


def pairwise_temporal_intersection_over_union(target_segments: np.ndarray, candidate_segments: np.ndarray) -> np.ndarray:
    r"""
    Computes the temporal intersection-over-union (tIoU) between all pairs of target and candidate segments.
    The IoU is also called the Jaccard index or the Jaccard similarity coefficient.

    The tIoU between a segment A and a segment B is: $\frac{|A \cup B|}{|A \cap B|}$.
    Args:
        target_segments: A Numpy array of shape (N, 2) containing N target segments (start, end).
        candidate_segments: A Numpy array of shape (M, 2) containing M candidate segments (start, end).

    Returns:
        iou_map: A Numpy array of size (N, M) containing the IoU score for all pairs of target and candidate segments.
    """
    return np.apply_along_axis(
        lambda segment: temporal_intersection_over_union(segment, candidate_segments),
        axis=1,
        arr=target_segments,
    )


def get_segment_positive_negative_count(tiou_map: np.ndarray, tiou_threshold: float):
    n_targets, n_proposals = tiou_map.shape
    matches = (tiou_map >= tiou_threshold)
    # Calculate true positives: targets with at least one matching proposal
    true_positives = matches.max(axis=1).sum()
    # Calculate false positives: proposals that did not match any target + excess predictions
    false_positives = n_proposals - matches.max(axis=0).sum() + np.maximum(matches.sum(axis=0) - 1, 0).sum()
    # Calculate false negatives: targets with no matching proposals
    false_negatives = n_targets - true_positives
    return true_positives, false_positives, false_negatives


def calculate_segment_prediction_metrics(tiou_map: np.ndarray, tiou_threshold: float = 0.5):
    if tiou_map.shape[1] == 0:
        return 0, 0
    true_positives, false_positives, false_negatives = get_segment_positive_negative_count(tiou_map, tiou_threshold)
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    return precision, recall
