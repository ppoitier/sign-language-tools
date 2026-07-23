"""Vertex indices for the 21-landmark MediaPipe hand model, grouped by finger.

Each constant is a tuple of indices into the `(21, C)` array of hand landmarks
produced by MediaPipe (see `results.left_hand_landmarks` / `right_hand_landmarks`
in [`extract_poses_from_video_file`][sign_language_tools.pose.mediapipe.extraction.extract_poses_from_video_file]).
These groups are typically used to select or mask a subset of the hand
landmarks, e.g. to isolate a single finger.
"""

__all__ = [
    "PALM_IDXS",
    "THUMB_IDXS",
    "INDEX_IDXS",
    "MIDDLE_IDXS",
    "RING_IDXS",
    "PINKY_IDXS",
]

PALM_IDXS = (0, 1, 5, 9, 13, 17)
"""Indices of the wrist and the base (MCP/CMC) joints of each finger."""

THUMB_IDXS = (2, 3, 4)
"""Indices of the thumb joints, from the CMC joint to the fingertip."""

INDEX_IDXS = (6, 7, 8)
"""Indices of the index finger joints, from the PIP joint to the fingertip."""

MIDDLE_IDXS = (10, 11, 12)
"""Indices of the middle finger joints, from the PIP joint to the fingertip."""

RING_IDXS = (14, 15, 16)
"""Indices of the ring finger joints, from the PIP joint to the fingertip."""

PINKY_IDXS = (18, 19, 20)
"""Indices of the pinky finger joints, from the PIP joint to the fingertip."""
