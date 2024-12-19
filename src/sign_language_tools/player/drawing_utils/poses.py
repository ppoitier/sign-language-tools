import cv2
import numpy as np

from sign_language_tools.player.drawing_utils.utils import to_frame_coords


def draw_pose(
    frame: np.ndarray,
    pose: np.ndarray,
    edges: np.ndarray | None,
    vertex_lims: np.ndarray,
    frame_lims: np.ndarray,
    vertex_color: tuple[int, int, int],
    edge_color: tuple[int, int, int],
    vertex_width: int,
    edge_width: int,
):
    positions = to_frame_coords(pose, vertex_lims, frame_lims)
    if edges is not None:
        for idx1, idx2 in edges:
            cv2.line(frame, positions[idx1], positions[idx2], edge_color, edge_width)
    for pos in positions:
        cv2.circle(frame, pos, vertex_width, vertex_color, -1, cv2.LINE_AA)
