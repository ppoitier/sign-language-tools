from sign_language_tools.core.transform import Transform


class Identity(Transform):
    """Returns its input(s) unchanged.

    Useful as a no-op placeholder wherever a transform is expected, e.g. as
    the "do nothing" branch of
    [`Randomize`][sign_language_tools.common.transforms.randomize.Randomize].

    Example:
        >>> from sign_language_tools.common.transforms import Identity
        >>> transform = Identity()
        >>> transform(42)
        42
    """

    def __call__(self, *args):
        """Returns the input(s) unchanged.

        Args:
            *args: Any number of inputs.

        Returns:
            `args[0]` if a single argument was given, otherwise the full
            `args` tuple.
        """
        if len(args) == 1:
            return args[0]
        return args
