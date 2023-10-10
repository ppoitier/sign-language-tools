from sign_language_tools.core.transform import Transform


class Identity(Transform):
    def __call__(self, *args):
        if len(args) == 1:
            return args[0]
        return args
