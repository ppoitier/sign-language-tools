import numpy as np

from sign_language_tools.core.transform import Transform


class NormalizeByReferenceEdge(Transform):
    """Scale a pose sequence using a reference edge as the scale unit.

    Args:
        ref_edge: Indices `(p1, p2)` of the two landmarks defining the reference edge.
        per_frame: If `True`, compute and apply a scaling factor independently for
            each frame, forcing the reference edge to length 1 in every frame.
            If `False` (default), compute a single scaling factor from the mean
            edge length over the whole sequence, which removes the sequence's
            overall scale while preserving relative scale changes across frames.
        eps: Small value added to the edge length to avoid division by zero.
    """

    def __init__(self, ref_edge: tuple[int, int], per_frame: bool = False, eps: float = 1e-8):
        super().__init__()
        self.ref_edge = ref_edge
        self.per_frame = per_frame
        self.eps = eps

    def __call__(self, pose_seq: np.ndarray) -> np.ndarray:
        p1_idx, p2_idx = self.ref_edge
        edge_lengths = np.linalg.norm(pose_seq[:, p2_idx] - pose_seq[:, p1_idx], axis=-1)  # Shape: (T,)

        if self.per_frame:
            scale = edge_lengths + self.eps  # Shape: (T,)
            return pose_seq / scale[:, None, None]

        scale = edge_lengths.mean() + self.eps  # Scalar
        return pose_seq / scale
