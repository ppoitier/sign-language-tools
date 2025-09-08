import numpy as np

from sign_language_tools.core.transform import Transform


class NormalizeEdgeLengths(Transform):
    def __init__(self, ref_edge: tuple[int, int]):
        super().__init__()
        self.edge = ref_edge

    def __call__(self, pose_seq: np.ndarray) -> np.ndarray:
        edge_lengths = np.linalg.norm(pose_seq[:, self.edge[1]] - pose_seq[:, self.edge[0]], axis=1).mean()
        return pose_seq / edge_lengths


class NormalizeToUnitaryEdge(Transform):
    def __init__(self, ref_edge: tuple[int, int]):
        super().__init__()
        self.ref_edge = ref_edge

    def __call__(self, pose_seq: np.ndarray) -> np.ndarray:
        p1_idx, p2_idx = self.ref_edge
        p1 = pose_seq[:, p1_idx, :]  # Shape: (T, 2)
        p2 = pose_seq[:, p2_idx, :]  # Shape: (T, 2)

        # 2. Calculate the length of the reference edge for each frame
        # We add a small epsilon to prevent division by zero for zero-length edges
        edge_lengths = np.linalg.norm(p2 - p1, axis=1) + 1e-8  # Shape: (T,)

        # 3. Calculate the scaling factor for each frame
        # We want to scale the pose so that edge_lengths becomes 1
        scaling_factors = 1.0 / edge_lengths

        # 4. Apply the scaling factor to all landmarks
        # We reshape scaling_factors to (T, 1, 1) to broadcast correctly
        # across the (T, L, 2) pose_seq array.
        return pose_seq * scaling_factors[:, None, None]
