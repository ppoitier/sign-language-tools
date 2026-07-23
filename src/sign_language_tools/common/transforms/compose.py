from typing import Callable

from sign_language_tools.core.transform import Transform


class Compose(Transform):
    """Chains a sequence of transforms into a single callable transform.

    Args:
        transforms: List of transforms to apply in order.
        multi_input: If `True`, each transform in the chain receives and
            returns multiple positional arguments (`args` is unpacked into,
            and re-packed out of, every transform). If `False`, only the
            first positional argument is used as input and is passed
            through the chain as a single value; any extra arguments are
            ignored.

    Example:
        >>> from sign_language_tools.common.transforms import Compose
        >>> transform = Compose([lambda x: x + 1, lambda x: x * 2])
        >>> transform(3)
        8
    """

    def __init__(self, transforms: list[Callable], multi_input: bool = False):
        super().__init__()
        self.transforms = transforms
        self.multi_input = multi_input

    def __call__(self, *args):
        """Applies the chained transforms in order.

        Args:
            *args: Input(s) to transform. If `multi_input` is `False`,
                only the first argument is used.

        Returns:
            The output of the last transform in the chain.
        """
        if self.multi_input:
            for t in self.transforms:
                args = t(*args)
        else:
            args = args[0]
            for t in self.transforms:
                args = t(args)
        return args
