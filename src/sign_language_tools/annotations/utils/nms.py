import numpy as np

from sign_language_tools.annotations.utils.iou import temporal_intersection_over_union


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
        remaining_proposals: A numpy array of shape (L, 3) that is a subset of the K initial proposals.
    """
    selected_proposals = []
    proposals = proposals[proposals[:, 2].argsort()[::-1]]
    while len(proposals) > 0:
        highest_score_proposal = proposals[0]
        selected_proposals.append(highest_score_proposal)
        ious = temporal_intersection_over_union(highest_score_proposal[:2], proposals[:, :2])
        proposals = proposals[ious < iou_threshold]
    return np.array(selected_proposals)


def soft_nms(proposals: np.ndarray, alpha: float = 0.4, t1: float = 0.5, t2: float = 0.9, n_proposals: int = 100):
    """
    Filter out a set of segment proposals using the Soft Non-Maximum Suppression (soft NMS) algorithm
    with a gaussian decay.

    Args:
        proposals: A Numpy array of shape (K, 3) containing K proposals with scores associated to them.
            Each proposal is in the form (start, end, score).
        alpha: Alpha value of Gaussian decaying function. Default = 0.4.
        t1: Minimum thresholds for soft NMS. Default = 0.5.
        t2: Maximum thresholds for soft NMS. Default = 0.9.
        n_proposals: Maximum number of retained proposals. Default = 100.

    Returns:
        remaining_proposals: A numpy array of shape (L, 3) that is a subset of the K initial proposals.
    """
    proposals = proposals[np.argsort(-proposals[:, 2])]

    start = list(proposals[:, 0])
    end = list(proposals[:, 1])
    score = list(proposals[:, 2])

    final_start = []
    final_end = []
    final_score = []

    while len(score) > 1 and len(final_score) < 101:
        max_index = score.index(max(score))
        iou_list = temporal_intersection_over_union(
            (start[max_index], end[max_index]),
            np.stack([start, end]).T,
        )

        for idx in range(0, len(score)):
            if idx != max_index:
                iou = iou_list[idx]
                tmp_width = end[max_index] - start[max_index]
                if iou > t1 + (t2 - t1) * tmp_width:
                    score[idx] *= np.exp(-np.square(iou) / alpha)

        final_start.append(start[max_index])
        final_end.append(end[max_index])
        final_score.append(score[max_index])
        start.pop(max_index)
        end.pop(max_index)
        score.pop(max_index)

    return np.stack([final_start, final_end, final_score]).T
