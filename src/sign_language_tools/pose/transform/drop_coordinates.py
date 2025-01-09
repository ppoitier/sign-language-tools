import numpy as np

from sign_language_tools.core.transform import Transform


class DropCoordinates(Transform):
    def __init__(self, coord: int | str):
        super().__init__()
        if isinstance(coord, str):
            coord = ['x', 'y', 'z'].index(coord)
        self.coord_indices = [i for i in range(3) if i != coord]

    def __call__(self, pose_seq: np.ndarray) -> np.ndarray:
        return pose_seq[:, :, self.coord_indices]
