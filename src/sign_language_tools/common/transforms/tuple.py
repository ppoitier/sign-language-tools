from sign_language_tools.core.transform import Transform


class TransformTuple(Transform):
    def __init__(self, transform: callable, n: int = 2):
        super().__init__()
        self.transform = transform
        self.n = n

    def __call__(self, *args, **kwargs):
        return tuple([self.transform(*args, **kwargs) for _ in range(self.n)])