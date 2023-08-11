from sign_language_tools.core.transform import Transform


class Compose(Transform):
    def __init__(self, transforms: list[callable], multi_input: bool = False):
        super().__init__()
        self.transforms = transforms
        self.multi_input = multi_input

    def __call__(self, *args):
        if self.multi_input:
            for t in self.transforms:
                args = t(*args)
        else:
            args = args[0]
            for t in self.transforms:
                args = t(args)
        return args
