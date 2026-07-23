from typing import Callable

from sign_language_tools.core.transform import Transform


class TransformTuple(Transform):
    """Applies a transform independently `n` times and returns the results as a tuple.

    Note:
        `transform` is called `n` times with the exact same input(s), so it
        only makes sense to use a stochastic transform (e.g.
        [`Randomize`][sign_language_tools.common.transforms.randomize.Randomize]
        or a transform involving random noise) here. With a deterministic
        transform, all `n` outputs would be identical.

    Args:
        transform: The transform to apply.
        n: Number of times to apply `transform`.

    Example:
        >>> import numpy as np
        >>> from sign_language_tools.common.transforms import TransformTuple
        >>> add_noise = lambda x: x + np.random.normal(size=x.shape)
        >>> transform = TransformTuple(add_noise, n=2)
        >>> views = transform(np.zeros(3))
        >>> len(views)
        2
    """

    def __init__(self, transform: Callable, n: int = 2):
        super().__init__()
        self.transform = transform
        self.n = n

    def __call__(self, *args, **kwargs):
        """Applies the transform `n` times.

        Args:
            *args: Positional arguments forwarded to `transform`.
            **kwargs: Keyword arguments forwarded to `transform`.

        Returns:
            A tuple of `n` outputs of `transform`.
        """
        return tuple([self.transform(*args, **kwargs) for _ in range(self.n)])
