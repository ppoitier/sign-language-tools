

class Compose:
    def __init__(self, transforms: list[callable]):
        self.transforms = transforms

    def __call__(self, x):
        for t in self.transforms:
            x = t(x)
        return x
