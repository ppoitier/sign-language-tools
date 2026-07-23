from typing import Callable

import numpy as np

from sign_language_tools.core.transform import Transform
from sign_language_tools.common.transforms import Identity


class Randomize(Transform):
    """Applies the given transform with a given probability.

    On each call, `transform` is applied with probability `probability`;
    otherwise the input(s) are passed through unchanged (see
    [`Identity`][sign_language_tools.common.transforms.identity.Identity]).

    Args:
        transform: The transform to apply.
        probability: Probability of applying `transform`, between 0 and 1.

    Example:
        >>> from sign_language_tools.common.transforms import Randomize
        >>> transform = Randomize(lambda x: x * 2, probability=1.0)
        >>> transform(3)
        6
    """

    def __init__(self, transform: Callable, probability: float = 0.5):
        super().__init__()
        self.transform = transform
        self.identity = Identity()
        self.probability = probability

    def __call__(self, *args):
        """Randomly applies the transform or the identity.

        Args:
            *args: Input(s) to pass to `transform` (or to return unchanged).

        Returns:
            The output of `transform` with probability `probability`,
            otherwise the input(s) unchanged.
        """
        if np.random.rand() < self.probability:
            return self.transform(*args)
        return self.identity(*args)
