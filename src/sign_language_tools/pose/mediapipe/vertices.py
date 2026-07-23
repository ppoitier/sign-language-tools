"""Vertex groups for the MediaPipe face mesh, derived from its edge topology.

Each constant below is a tuple of landmark indices into the `(478, C)` array of
face landmarks produced by MediaPipe (see `results.face_landmarks` in
[`extract_poses_from_video_file`][sign_language_tools.pose.mediapipe.extraction.extract_poses_from_video_file]).

The vertex tuples are computed from the corresponding `FACEMESH_*` edge
definitions in `sign_language_tools.pose.mediapipe.facemesh` rather than
listed by hand, so they always stay in sync with the edges: any vertex that
appears in an edge is included, and duplicates are removed.

"Left"/"right" follow MediaPipe's convention, i.e. the subject's own left and
right, not the viewer's.
"""

from sign_language_tools.pose.mediapipe.facemesh import (
    FACEMESH_LIPS,
    FACEMESH_LEFT_EYE,
    FACEMESH_RIGHT_EYE,
    FACEMESH_LEFT_IRIS,
    FACEMESH_RIGHT_IRIS,
    FACEMESH_LEFT_EYEBROW,
    FACEMESH_RIGHT_EYEBROW,
)


__all__ = [
    "LIPS_VERTICES",
    "LEFT_EYE_VERTICES",
    "RIGHT_EYE_VERTICES",
    "LEFT_IRIS_VERTICES",
    "RIGHT_IRIS_VERTICES",
    "LEFT_EYEBROW_VERTICES",
    "RIGHT_EYEBROW_VERTICES",
    "LEFT_FULL_EYE_VERTICES",
    "RIGHT_FULL_EYE_VERTICES",
]


def _vertices_from_edges(edges: tuple[tuple[int, int], ...]) -> tuple[int, ...]:
    """Collect the unique, sorted vertex indices referenced by a set of edges.

    Args:
        edges (tuple[tuple[int, int], ...]): Edges as `(start_index, end_index)` pairs.

    Returns:
        tuple[int, ...]: Sorted tuple of the distinct vertex indices found in `edges`.
    """
    return tuple(sorted(set(sum(edges, ()))))


LIPS_VERTICES = _vertices_from_edges(FACEMESH_LIPS)
"""Vertex indices of the lips contour."""

LEFT_EYE_VERTICES = _vertices_from_edges(FACEMESH_LEFT_EYE)
"""Vertex indices of the left eye contour."""

RIGHT_EYE_VERTICES = _vertices_from_edges(FACEMESH_RIGHT_EYE)
"""Vertex indices of the right eye contour."""

LEFT_IRIS_VERTICES = _vertices_from_edges(FACEMESH_LEFT_IRIS)
"""Vertex indices of the left iris."""

RIGHT_IRIS_VERTICES = _vertices_from_edges(FACEMESH_RIGHT_IRIS)
"""Vertex indices of the right iris."""

LEFT_EYEBROW_VERTICES = _vertices_from_edges(FACEMESH_LEFT_EYEBROW)
"""Vertex indices of the left eyebrow."""

RIGHT_EYEBROW_VERTICES = _vertices_from_edges(FACEMESH_RIGHT_EYEBROW)
"""Vertex indices of the right eyebrow."""

LEFT_FULL_EYE_VERTICES = LEFT_EYE_VERTICES + LEFT_IRIS_VERTICES + LEFT_EYEBROW_VERTICES
"""Vertex indices of the left eye, iris and eyebrow combined."""

RIGHT_FULL_EYE_VERTICES = RIGHT_EYE_VERTICES + RIGHT_IRIS_VERTICES + RIGHT_EYEBROW_VERTICES
"""Vertex indices of the right eye, iris and eyebrow combined."""
