from sign_language_tools.features.landmarks.edges.facemesh import (
    FACEMESH_CONTOURS,
    FACEMESH_TESSELATION,
)
from sign_language_tools.features.landmarks.edges.base import (
    POSE_EDGES,
    HAND_EDGES,
)
from sign_language_tools.visualisation.video.cv.utils import draw_connections, draw_points


def draw_landmarks(
        img,
        landmarks,
        connections=None,
        edge_color=(0, 255, 0),
        vertex_color=(0, 0, 255),
        show_vertices: bool = True,
):
    if connections is not None:
        draw_connections(img, landmarks, connections, edge_color)
    if show_vertices:
        draw_points(img, landmarks, vertex_color)


def draw_pose_landmarks(img, positions):
    draw_connections(img, positions, POSE_EDGES, color=(0, 255, 0))
    draw_points(img, positions)


def draw_hands_landmarks(img, positions):
    draw_hand_landmarks(img, positions[:21], color=(255, 0, 0))
    draw_hand_landmarks(img, positions[21:])


def draw_hand_landmarks(img, positions, color=(0, 0, 255)):
    draw_connections(img, positions, HAND_EDGES, color)


def draw_face_landmarks(img, positions, color=(0, 255, 0)):
    draw_connections(img, positions, FACEMESH_CONTOURS, color, thickness=1)
    draw_points(img, positions, color=(0, 0, 255), radius=1)


def draw_face_mesh(img, positions, color=(0, 255, 0)):
    draw_connections(img, positions, FACEMESH_TESSELATION, color, thickness=1)
