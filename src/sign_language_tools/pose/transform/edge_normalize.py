import numpy as np

from sign_language_tools.core.transform import Transform


class NormalizeEdgeLengths(Transform):
    def __init__(self, unitary_edge: tuple[int, int]):
        super().__init__()
        self.edge = unitary_edge

    def __call__(self, pose_seq: np.ndarray) -> np.ndarray:
        edge_lengths = np.linalg.norm(pose_seq[:, self.edge[1]] - pose_seq[:, self.edge[0]], axis=1).mean()
        return pose_seq / edge_lengths
