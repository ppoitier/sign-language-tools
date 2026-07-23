import numpy as np

from sign_language_tools.core.transform import Transform


class Concatenate(Transform):
    """Concatenates a sequence of arrays along a given axis.

    Args:
        dim: Axis along which the arrays are concatenated.

    Example:
        >>> import numpy as np
        >>> from sign_language_tools.common.transforms import Concatenate
        >>> transform = Concatenate(dim=0)
        >>> transform((np.zeros((2, 3)), np.zeros((4, 3)))).shape
        (6, 3)
    """

    def __init__(self, dim: int = 0):
        super().__init__()
        self.dim = dim

    def __call__(self, x: tuple[np.ndarray, ...]) -> np.ndarray:
        """Concatenates the given arrays.

        Args:
            x: Sequence of arrays to concatenate. All arrays must have the
                same shape, except along `dim`.

        Returns:
            The arrays concatenated along `dim`.
        """
        return np.concatenate(x, axis=self.dim)
