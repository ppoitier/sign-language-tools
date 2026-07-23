"""Edge (connectivity) definitions for MediaPipe hand, pose and face landmarks.

Each constant is a tuple of `(start_index, end_index)` pairs, where the indices
refer to positions in the corresponding landmark array (hand: `(21, C)`, pose:
`(33, C)`, face: `(478, C)`). These are typically used to draw skeletons or
build adjacency structures (e.g. graph-based models) over the landmarks.

"Upper"/"lower" pose edges split the body pose skeleton into the upper body
(arms, shoulders, face landmarks 0-10) and the lower body (hips and legs).
"""

from sign_language_tools.pose.mediapipe.facemesh import FACEMESH_CONTOURS


__all__ = [
    "LIPS_EDGES",
    "EYE_EDGES",
    "IRIS_EDGES",
    "EYEBROW_EDGES",
    "FULL_EYE_EDGES",
    "HAND_EDGES",
    "UPPER_POSE_EDGES",
    "LOWER_POSE_EDGES",
    "POSE_EDGES",
    "FACE_EDGES",
]


LIPS_EDGES = (
    (7, 17),
    (17, 15),
    (15, 19),
    (19, 12),
    (12, 3),
    (3, 30),
    (30, 37),
    (37, 33),
    (33, 35),
    (35, 25),
    (7, 20),
    (20, 6),
    (6, 5),
    (5, 4),
    (4, 0),
    (0, 22),
    (22, 23),
    (23, 24),
    (24, 38),
    (38, 25),
    (8, 16),
    (16, 14),
    (14, 18),
    (18, 13),
    (13, 2),
    (2, 31),
    (31, 36),
    (36, 32),
    (32, 34),
    (34, 26),
    (8, 21),
    (21, 9),
    (9, 10),
    (10, 11),
    (11, 1),
    (1, 29),
    (29, 28),
    (28, 27),
    (27, 39),
    (39, 26),
)
"""Edges of the lips contour, over the `LIPS_VERTICES`-scale local indexing
(not the raw MediaPipe face mesh indices)."""

EYE_EDGES = (
    (1, 0),
    (0, 13),
    (13, 3),
    (3, 4),
    (4, 5),
    (5, 6),
    (6, 7),
    (7, 2),
    (1, 15),
    (15, 12),
    (12, 11),
    (11, 10),
    (10, 9),
    (9, 8),
    (8, 14),
    (14, 2),
)
"""Edges of a single eye contour, over the `LEFT_EYE_VERTICES`/`RIGHT_EYE_VERTICES`-scale
local indexing (not the raw MediaPipe face mesh indices). Shared by both eyes since they
have the same local topology."""

IRIS_EDGES = ((0, 1), (1, 2), (2, 3), (3, 0))
"""Edges of a single iris contour, over the `LEFT_IRIS_VERTICES`/`RIGHT_IRIS_VERTICES`-scale
local indexing (not the raw MediaPipe face mesh indices). Shared by both irises."""

EYEBROW_EDGES = ((0, 2), (2, 1), (1, 5), (5, 3), (7, 4), (4, 8), (8, 6), (6, 9))
"""Edges of a single eyebrow contour, over the `LEFT_EYEBROW_VERTICES`/`RIGHT_EYEBROW_VERTICES`-scale
local indexing (not the raw MediaPipe face mesh indices). Shared by both eyebrows."""

FULL_EYE_EDGES = (
    (1, 0),
    (0, 13),
    (13, 3),
    (3, 4),
    (4, 5),
    (5, 6),
    (6, 7),
    (7, 2),
    (1, 15),
    (15, 12),
    (12, 11),
    (11, 10),
    (10, 9),
    (9, 8),
    (8, 14),
    (14, 2),
    (16, 17),
    (17, 18),
    (18, 19),
    (19, 16),
    (20, 22),
    (22, 21),
    (21, 25),
    (25, 23),
    (27, 24),
    (24, 28),
    (28, 26),
    (26, 29),
)
"""Edges of a single eye, iris and eyebrow combined, over the
`LEFT_FULL_EYE_VERTICES`/`RIGHT_FULL_EYE_VERTICES`-scale local indexing (not the raw
MediaPipe face mesh indices). Shared by both sides."""

HAND_EDGES = (
    (3, 4),
    (0, 5),
    (17, 18),
    (0, 17),
    (13, 14),
    (13, 17),
    (18, 19),
    (5, 6),
    (5, 9),
    (14, 15),
    (0, 1),
    (9, 10),
    (1, 2),
    (9, 13),
    (10, 11),
    (19, 20),
    (6, 7),
    (15, 16),
    (2, 3),
    (11, 12),
    (7, 8),
)
"""Edges of the 21-landmark hand skeleton (wrist and the four joints of each finger),
using the raw MediaPipe hand landmark indices. Shared by both hands."""

UPPER_POSE_EDGES = (
    (15, 21),
    (16, 20),
    (18, 20),
    (3, 7),
    (14, 16),
    (6, 8),
    (15, 17),
    (16, 22),
    (4, 5),
    (5, 6),
    (0, 1),
    (9, 10),
    (1, 2),
    (0, 4),
    (11, 13),
    (15, 19),
    (16, 18),
    (12, 14),
    (17, 19),
    (2, 3),
    (11, 12),
    (13, 15),
)
"""Edges of the upper body: face (nose, eyes, ears, mouth), shoulders and arms, using
the raw MediaPipe pose landmark indices (0-22)."""

LOWER_POSE_EDGES = (
    (11, 23),
    (12, 24),
    (23, 24),
    (23, 25),
    (24, 26),
    (25, 27),
    (26, 28),
    (27, 29),
    (28, 30),
    (29, 31),
    (30, 32),
    (27, 31),
    (28, 32),
)
"""Edges of the lower body: hips and legs, using the raw MediaPipe pose landmark
indices (11-32)."""

POSE_EDGES = LOWER_POSE_EDGES + UPPER_POSE_EDGES
"""Full body pose skeleton: `LOWER_POSE_EDGES` and `UPPER_POSE_EDGES` combined."""

FACE_EDGES = FACEMESH_CONTOURS
"""Face contours (lips, eyes, eyebrows and face outline), using the raw MediaPipe
face mesh indices. Alias of [`FACEMESH_CONTOURS`][sign_language_tools.pose.mediapipe.facemesh.FACEMESH_CONTOURS]."""
