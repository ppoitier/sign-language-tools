import numpy as np

from sign_language_tools.pose.transform.functional.rotation import rotation_matrix_y, rotation_matrix_x
from sign_language_tools.core.transform import Transform


class RandomRotation3D(Transform):
    def __init__(
            self,
            mode: str = 'horizontal',
            angle_range: tuple[float, float] = (-np.pi/6, np.pi/6),
            rotation_center: tuple[float, float, float] = (0.5, 0.5, 1.0),
    ):
        super().__init__()
        self.mode = mode
        self.angle_range = angle_range
        self.rot_center = np.array(rotation_center)

    def __call__(self, pose_sequence: np.ndarray):
        angle = np.random.uniform(*self.angle_range)
        if self.mode == 'horizontal':
            rot_matrix = rotation_matrix_y(angle)
        else:
            rot_matrix = rotation_matrix_x(angle)

        T, L, C = pose_sequence.shape
        if C == 2:
            extended_data = np.ones((T, L, 3))
            extended_data[:, :, :C] = pose_sequence
            pose_sequence = extended_data

        pose_sequence -= self.rot_center
        pose_sequence = pose_sequence @ rot_matrix
        pose_sequence += self.rot_center

        return pose_sequence[:, :, :C]
