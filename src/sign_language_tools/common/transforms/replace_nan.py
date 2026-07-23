import numpy as np

from sign_language_tools.core.transform import Transform


class ReplaceNaN(Transform):
    """Replaces `NaN` values in an array with a fixed value.

    Warning:
        This transform mutates `x` in place (in addition to returning it),
        since it assigns directly into the input array.

    Args:
        fill_value: Value used to replace `NaN` entries.

    Example:
        >>> import numpy as np
        >>> from sign_language_tools.common.transforms import ReplaceNaN
        >>> transform = ReplaceNaN(fill_value=0.0)
        >>> transform(np.array([1.0, np.nan, 3.0]))
        array([1., 0., 3.])
    """

    def __init__(self, fill_value: float = 0.0):
        super().__init__()
        self.fill_value = fill_value

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Replaces the `NaN` values of `x` with `fill_value`.

        Args:
            x: Array of floats, potentially containing `NaN` values.

        Returns:
            `x`, with every `NaN` entry replaced by `fill_value`.
        """
        x[np.isnan(x)] = self.fill_value
        return x
