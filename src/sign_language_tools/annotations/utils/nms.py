import numpy as np

from sign_language_tools.annotations.utils.iou import intersection_over_union


def non_maximum_suppression(proposals: np.ndarray, iou_threshold: float = 0.8) -> np.ndarray:
    """
    Filter out a set of segment proposals using the Non-Maximum Suppression (NMS) algorithm.
    A high scoring proposal is selected, then only proposals with a IoU score less than a given threshold are kept.
    This operation removes proposals that overlap too much.

    Args:
        proposals: A Numpy array of shape (K, 3) containing K proposals with scores associated to them.
            Each proposal is in the form (start, end, score).
        iou_threshold: The IoU threshold used to filter out proposals that overlap too much.

    Returns:
        remaining_proposals: A subset of the K initial proposals.
    """
    selected_proposals = []
    proposals = proposals[proposals[:, 2].argsort()[::-1]]
    while len(proposals) > 0:
        highest_score_proposal = proposals[0]
        selected_proposals.append(highest_score_proposal)
        ious = intersection_over_union(highest_score_proposal[:2], proposals[:, :2])
        proposals = proposals[ious < iou_threshold]
    return np.array(selected_proposals)
