import numpy as np

from sign_language_tools.core.transform import Transform


class DropCoordinates(Transform):
    """Drops one of the x, y, z coordinates from a pose sequence.

    Args:
        coord: Coordinate to drop, either as an index (`0`, `1`, `2`) or as
            a name (`"x"`, `"y"`, `"z"`).

    Example:
        >>> import numpy as np
        >>> from sign_language_tools.pose.transforms import DropCoordinates
        >>> pose_sequence = np.random.rand(10, 33, 3)  # (T, L, C)
        >>> transform = DropCoordinates(coord="z")
        >>> transform(pose_sequence).shape
        (10, 33, 2)
    """

    def __init__(self, coord: int | str):
        super().__init__()
        if isinstance(coord, str):
            coord = ['x', 'y', 'z'].index(coord)
        self.coord_indices = [i for i in range(3) if i != coord]

    def __call__(self, pose_sequence: np.ndarray) -> np.ndarray:
        """Drops the configured coordinate from the pose sequence.

        Args:
            pose_sequence: Pose sequence of shape `(T, L, 3)`, where `T` is
                the number of frames and `L` the number of landmarks.

        Returns:
            The pose sequence without the dropped coordinate, of shape
            `(T, L, 2)`.
        """
        return pose_sequence[:, :, self.coord_indices]
