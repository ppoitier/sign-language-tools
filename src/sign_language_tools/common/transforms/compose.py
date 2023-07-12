from sign_language_tools.core.transform import Transform


class Compose(Transform):
    def __init__(self, transforms: list[callable]):
        super().__init__()
        self.transforms = transforms

    def __call__(self, *args):
        for t in self.transforms:
            args = t(*args)
        return args
