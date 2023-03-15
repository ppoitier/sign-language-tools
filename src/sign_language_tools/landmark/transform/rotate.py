import numpy as np
import random
from math import cos, sin, pi


class Rotation2D:

    def __init__(self, angle: float, center=(0.5, 0.5)):
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
    def __init__(self, angle_range=(0, 2*pi), center=(0.5, 0.5)):
        r0, r1 = angle_range
        angle = random.uniform(r0, r1)
        super().__init__(angle, center)
