import numpy as np
import random
from math import cos, sin, pi

from sign_language_tools.core.transform import Transform


class Rotation2D(Transform):
    def __init__(self, angle: float, center=(0.5, 0.5)):
        super().__init__()
        self.center = center
        self.angle = angle

    def __call__(self, landmarks: np.ndarray):
        cx, cy = self.center
        r = self.angle
        x = landmarks[:, :, 0].copy() - cx
        y = landmarks[:, :, 1].copy() - cy
        landmarks[:, :, 0] = x * cos(r) - y * sin(r) + cx
        landmarks[:, :, 1] = x * sin(r) + y * cos(r) + cy
        return landmarks


class RandomRotation2D(Rotation2D):
    def __init__(self, angle_range=(-pi/12, pi/12), center=(0.5, 0.5)):
        self.angle_range = angle_range

        r0, r1 = angle_range
        angle = random.uniform(r0, r1)
        super().__init__(angle, center)

    def __call__(self, landmarks: np.ndarray):
        self.angle = random.uniform(*self.angle_range)
        return super().__call__(landmarks)


class MakeReferenceEdgeHorizontal(Transform):
    """
    Rotates the pose in each frame so the reference edge is horizontal.
    The rotation aligns the reference vector with the positive x-axis.

    This transform only works with pose sequences of shape (T, L, 2)
    where T is the number of frames and L the number of landmarks.
    """

    def __init__(self, ref_idx: tuple[int, int]):
        super().__init__()
        if not (isinstance(ref_idx, (tuple, list)) and len(ref_idx) == 2):
            raise ValueError("Reference indices must be a tuple of two integers.")
        self.ref_idx = ref_idx

    def __call__(self, pose_sequence: np.ndarray) -> np.ndarray:
        """
        Applies the rotation to a pose sequence.

        Args:
            pose_sequence (np.ndarray): A numpy array of shape (T, L, 2).

        Returns:
            np.ndarray: The rotated pose sequence of shape (T, L, 2).
        """
        pose_sequence = pose_sequence.copy()
        p1_idx, p2_idx = self.ref_idx

        # 1. Calculate the angle of the reference edge for each frame
        # Vector from p1 to p2 for all frames T
        ref_vec = pose_sequence[:, p2_idx, :] - pose_sequence[:, p1_idx, :]
        print(ref_vec.shape)
        # Angle of each vector relative to the positive x-axis
        ref_angles = np.arctan2(ref_vec[:, 1], ref_vec[:, 0])
        print(ref_angles.shape)

        # 2. Build the rotation matrices to cancel out these angles
        # We need to rotate by -angles
        cos_a = np.cos(-ref_angles)
        sin_a = np.sin(-ref_angles)

        # Create a stack of rotation matrices, one for each frame
        # Shape: (T, 2, 2)
        rot_mats = np.zeros((pose_sequence.shape[0], 2, 2))
        rot_mats[:, 0, 0] = cos_a
        rot_mats[:, 0, 1] = sin_a
        rot_mats[:, 1, 0] = -sin_a
        rot_mats[:, 1, 1] = cos_a

        # 3. Rotate around a pivot point (the first reference landmark)
        # This ensures the rotation doesn't change the hand's position in space.
        pivots = pose_sequence[:, p1_idx, :][:, None, :]  # Shape: (T, 1, 2)

        # Translate to origin, rotate, then translate back
        translated_pose = pose_sequence - pivots
        # Apply batched matrix multiplication using einsum
        rotated_pose = np.einsum("tlj,tjk->tlk", translated_pose, rot_mats)

        return rotated_pose + pivots
