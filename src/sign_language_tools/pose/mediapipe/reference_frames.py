"""Reference frame alignments for MediaPipe landmark groups.

MediaPipe does not measure every body part from the same origin, which matters as
soon as the depth is used — to draw the groups in one 3D scene, or to compute on
them. Each constant here describes, for one kind of MediaPipe output, how to
bring the groups into a single frame; pass it to
[`align_reference_frames`][sign_language_tools.pose.reference_frames.align_reference_frames]
or straight to
[`plot_pose_3d`][sign_language_tools.pose.visualization.plotly.graph_3d.plot_pose_3d].

The landmark indices used below are the raw MediaPipe pose indices: `0` is the
nose, `15` the left wrist and `16` the right wrist. On the face mesh side, `1` is
the nose tip.
"""

from sign_language_tools.pose.reference_frames import SharedLandmark


__all__ = [
    "MEDIAPIPE_DEPTH_ALIGNMENT",
    "MEDIAPIPE_WORLD_ALIGNMENT",
]


MEDIAPIPE_DEPTH_ALIGNMENT = {
    "left_hand": SharedLandmark("pose", 15, 0, axes=(2,)),
    "right_hand": SharedLandmark("pose", 16, 0, axes=(2,)),
    "face": SharedLandmark("pose", 0, 1, axes=(2,)),
}
"""Aligns the *image* landmarks (`pose_landmarks`, `hand_landmarks`,
`face_landmarks`), as returned by
[`extract_poses_from_video_file`][sign_language_tools.pose.mediapipe.extraction.extract_poses_from_video_file].

There, `x` and `y` are normalized to the image and therefore already share a
frame, but `z` is not: the depth of the pose is measured from the midpoint of the
hips, the depth of each hand from that hand's own wrist, and the depth of the
face from the head. Left as-is, the hands sit at the depth of the hips instead of
in front of the chest.

This alignment corrects **the depth only**, so the `x` and `y` you already trust
are left untouched. Each hand is moved onto the corresponding pose wrist, and the
face onto the pose nose."""

MEDIAPIPE_WORLD_ALIGNMENT = {
    "left_hand": SharedLandmark("pose", 15, 0),
    "right_hand": SharedLandmark("pose", 16, 0),
    "face": SharedLandmark("pose", 0, 1),
}
"""Aligns the *world* landmarks (`pose_world_landmarks`, `hand_world_landmarks`),
which are real-world coordinates in meters.

There, no axis is shared: the pose is centered between the hips and each hand is
centered on itself. The groups are therefore translated on **all three axes**,
rather than on the depth only."""
