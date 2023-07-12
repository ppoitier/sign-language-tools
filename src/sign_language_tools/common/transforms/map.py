from typing import Union

from sign_language_tools.core.transform import Transform


class MapTransform(Transform):
    def __init__(self, transforms: Union[list[callable], dict[any, callable]]):
        super().__init__()
        self.transforms = transforms

    def __call__(self, *args):
        if len(args) > 1:
            x = args
        else:
            x = args[0]

        if isinstance(x, list) or isinstance(x, tuple):
            self.transforms: list
            return tuple([
                transform(xx) if transform is not None else xx
                for xx, transform in zip(x, self.transforms)
            ])

        if isinstance(x, dict):
            self.transforms: dict
            for key in list(self.transforms.keys()):
                transform = self.transforms[key]
                if transform is not None:
                    x[key] = transform(x[key])
            return x

        raise ValueError("MapTransform only handles tuples, lists and dictionaries.")


class ApplyToAll(Transform):
    def __init__(self, transform: callable):
        super().__init__()
        self.transform = transform

    def __call__(self, x):
        if isinstance(x, dict):
            for key in list(x.keys()):
                x[key] = self.transform(x[key])
            return x
        if isinstance(x, list):
            return [self.transform(xx) for xx in x]