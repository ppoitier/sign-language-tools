"""Bringing landmark groups into a common reference frame.

Pose estimators rarely measure every body part from the same origin. MediaPipe,
for instance, measures the depth of the body from the midpoint of the hips but
the depth of each hand from that hand's own wrist, so the groups cannot be
compared — or drawn in a single 3D scene — until they are aligned.

Alignment is described by declaring which landmarks are physically the same
point across two groups, see [`SharedLandmark`][sign_language_tools.pose.reference_frames.SharedLandmark].
Ready-made descriptions for MediaPipe live in
[`sign_language_tools.pose.mediapipe.reference_frames`][sign_language_tools.pose.mediapipe.reference_frames].
"""

from typing import Mapping, NamedTuple

import numpy as np


__all__ = [
    "SharedLandmark",
    "align_reference_frames",
]


class SharedLandmark(NamedTuple):
    """Declares that a landmark of one group is the same point as one of another.

    This is all that is needed to place a group in another group's reference
    frame: if the two landmarks are the same physical point, the offset between
    them is exactly the offset between the two frames.

    Attributes:
        group: Name of the group to align on, e.g. `"pose"`.
        landmark: Index, within `group`, of the shared landmark.
        own_landmark: Index, within the group being moved, of the shared
            landmark. Defaults to `0`.
        axes: Coordinate axes the alignment applies to, as indices into the last
            dimension of the arrays. Defaults to all three axes; use `(2,)` to
            correct the depth only and leave `x` and `y` untouched.

    Example:
        The wrist of a hand is landmark `0` of that hand, and landmark `16` of a
        MediaPipe body pose:

        >>> from sign_language_tools.pose.reference_frames import SharedLandmark
        >>> SharedLandmark(group="pose", landmark=16, own_landmark=0, axes=(2,))
        SharedLandmark(group='pose', landmark=16, own_landmark=0, axes=(2,))
    """

    group: str
    landmark: int
    own_landmark: int = 0
    axes: tuple[int, ...] = (0, 1, 2)


def align_reference_frames(
    landmarks: Mapping[str, np.ndarray],
    alignment: Mapping[str, SharedLandmark],
) -> dict[str, np.ndarray]:
    """Translates landmark groups so that they share a common reference frame.

    Each group listed in `alignment` is translated so that its shared landmark
    coincides with the landmark it is aligned on. Groups keep their shape: the
    translation is rigid, and only moves a group as a whole.

    Groups that are not listed are returned unchanged, and so are groups whose
    shared landmark is missing (`NaN`) in this frame, since there is then nothing
    to align on.

    Args:
        landmarks: Mapping from a group name to its landmarks, of shape `(L, C)`.
        alignment: Mapping from the name of a group to the
            [`SharedLandmark`][sign_language_tools.pose.reference_frames.SharedLandmark]
            that places it. See
            [`MEDIAPIPE_DEPTH_ALIGNMENT`][sign_language_tools.pose.mediapipe.reference_frames.MEDIAPIPE_DEPTH_ALIGNMENT]
            for the usual MediaPipe case.

    Returns:
        A new mapping with the same keys as `landmarks`, holding the translated
        groups.

    Raises:
        ValueError: If the alignment names a group that is missing from
            `landmarks`, or refers to a landmark that does not exist.

    Example:
        >>> import numpy as np
        >>> from sign_language_tools.pose.reference_frames import align_reference_frames
        >>> from sign_language_tools.pose.mediapipe.reference_frames import (
        ...     MEDIAPIPE_DEPTH_ALIGNMENT,
        ... )
        >>> landmarks = {"pose": np.zeros((33, 3)), "right_hand": np.ones((21, 3))}
        >>> aligned = align_reference_frames(landmarks, MEDIAPIPE_DEPTH_ALIGNMENT)
        >>> hand = aligned["right_hand"]
        >>> hand[0]  # the wrist depth now matches the pose wrist
        array([1., 1., 0.])
    """
    aligned = dict(landmarks)
    for name, shared in alignment.items():
        if name not in aligned:
            continue
        if shared.group not in aligned:
            raise ValueError(
                f"Group '{name}' is aligned on group '{shared.group}',"
                " which is missing from the landmarks."
            )

        group = aligned[name]
        target = aligned[shared.group]
        _check_landmark(shared.own_landmark, len(group), name)
        _check_landmark(shared.landmark, len(target), shared.group)

        offset = np.zeros(group.shape[-1])
        axes = [axis for axis in shared.axes if axis < group.shape[-1]]
        offset[axes] = target[shared.landmark, axes] - group[shared.own_landmark, axes]

        aligned[name] = group + np.nan_to_num(offset)
    return aligned


def _check_landmark(index: int, landmark_count: int, name: str) -> None:
    if index >= landmark_count:
        raise ValueError(
            f"The alignment refers to landmark {index} of group '{name}',"
            f" which only has {landmark_count} landmarks."
        )
