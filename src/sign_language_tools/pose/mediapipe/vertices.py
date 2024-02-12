from sign_language_tools.pose.mediapipe.facemesh import FACEMESH_CONTOURS
from sign_language_tools.pose.mediapipe.edges import *

__all__ = [
    'LIPS_VERTICES',
    'HAND_VERTICES',
    'UPPER_POSE_VERTICES',
    'LOWER_POSE_VERTICES',
    'POSE_VERTICES',
    'FACE_VERTICES',
    'FACEMESH_CONTOURS_VERTICES',
]


def _vertices_from_edges(edges):
    return set(sum(edges, ()))


LIPS_VERTICES = _vertices_from_edges(LIPS_EDGES)

HAND_VERTICES = _vertices_from_edges(HAND_EDGES)

UPPER_POSE_VERTICES = _vertices_from_edges(UPPER_POSE_EDGES)

LOWER_POSE_VERTICES = _vertices_from_edges(LOWER_POSE_EDGES)

POSE_VERTICES = _vertices_from_edges(POSE_EDGES)

FACE_VERTICES = _vertices_from_edges(FACE_EDGES)

FACEMESH_CONTOURS_VERTICES = _vertices_from_edges(FACEMESH_CONTOURS)
