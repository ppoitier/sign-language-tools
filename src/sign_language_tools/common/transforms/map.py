from typing import Union, Callable, Any

from sign_language_tools.core.transform import Transform


class MapTransform(Transform):
    """Applies a different transform to each element of a tuple, list, or dict.

    Args:
        transforms: If `x` is a list/tuple, a list of transforms applied
            positionally (`transforms[i]` to `x[i]`). If `x` is a dict, a
            dict of transforms applied by key (`transforms[key]` to
            `x[key]`); keys of `x` absent from `transforms` are left
            untouched. In both cases, a `None` entry leaves the
            corresponding element unchanged.

    Warning:
        For list/tuple input, `transforms` and `x` are paired positionally
        with `zip`, so if `transforms` is shorter than `x`, the extra
        trailing elements of `x` are silently dropped from the output
        rather than passed through unchanged.

    Example:
        >>> from sign_language_tools.common.transforms import MapTransform
        >>> transform = MapTransform([lambda x: x + 1, None])
        >>> transform((1, 2))
        (2, 2)
    """

    def __init__(self, transforms: Union[list[Callable], dict[Any, Callable]]):
        super().__init__()
        self.transforms = transforms

    def __call__(self, *args):
        """Applies the mapped transforms to the input.

        Args:
            *args: A single tuple, list, or dict, or several positional
                arguments that are treated as a tuple.

        Returns:
            A tuple with each element mapped through the corresponding
            transform (if `x` was a tuple/list), or a dict with each value
            mapped through the corresponding transform (if `x` was a dict).

        Raises:
            ValueError: If the input is not a tuple, list, or dict.
        """
        if len(args) > 1:
            x = args
        else:
            x = args[0]

        if isinstance(x, list) or isinstance(x, tuple):
            self.transforms: list
            return tuple(
                [
                    transform(xx) if transform is not None else xx
                    for xx, transform in zip(x, self.transforms)
                ]
            )

        if isinstance(x, dict):
            x = dict(**x)
            self.transforms: dict
            for key in list(self.transforms.keys()):
                transform = self.transforms[key]
                if transform is not None:
                    x[key] = transform(x[key])
            return x

        raise ValueError("MapTransform only handles tuples, lists and dictionaries.")


class ApplyToAll(Transform):
    """Applies the same transform to every element of a list or dict.

    Args:
        transform: The transform applied to each element of `x`.

    Example:
        >>> from sign_language_tools.common.transforms import ApplyToAll
        >>> transform = ApplyToAll(lambda x: x + 1)
        >>> transform([1, 2, 3])
        [2, 3, 4]
    """

    def __init__(self, transform: Callable):
        super().__init__()
        self.transform = transform

    def __call__(self, x: Union[list, dict]) -> Union[list, dict]:
        """Applies the transform to every element of `x`.

        Args:
            x: A list or dict whose elements (or values) are transformed.

        Returns:
            A list, or dict, with `transform` applied to every element.

        Raises:
            ValueError: If `x` is not a list or dict.
        """
        if isinstance(x, dict):
            for key in list(x.keys()):
                x[key] = self.transform(x[key])
            return x
        if isinstance(x, list):
            return [self.transform(xx) for xx in x]
        raise ValueError("ApplyToAll only handles lists and dictionaries.")
