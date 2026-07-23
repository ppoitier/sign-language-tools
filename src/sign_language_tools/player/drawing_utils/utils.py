import numpy as np


def to_frame_coords(
    base_pos: np.ndarray,
    base_lims: np.ndarray,
    frame_lims: np.ndarray,
):
    """

    Args:
        base_pos: array of shape (N, ..., 2) containing positions in the base coordinate system.
        base_lims: array of shape (2, 2) containing the x_lim and y_lim of the base coordinate system.
        frame_lims: array of shape (2, 2) containing the x_lim and y_lim of the frame coordinate system.

    Returns:
        frame_pos: array of shape (N, 2) containing N positions in the frame coordinate system.
    """
    # Normalize the base positions
    shape = base_pos.shape
    base_pos = base_pos.reshape(-1, 2)
    base_range = base_lims[:, 1] - base_lims[:, 0]
    normalized_pos = (base_pos - base_lims[:, 0]) / base_range

    # Scale to frame coordinates
    frame_range = frame_lims[:, 1] - frame_lims[:, 0]
    frame_pos = normalized_pos * frame_range + frame_lims[:, 0]

    return frame_pos.reshape(shape).round().astype("int32")
