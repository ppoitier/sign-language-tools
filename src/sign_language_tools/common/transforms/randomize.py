import numpy as np

from sign_language_tools.core.transform import Transform
from sign_language_tools.common.transforms import Identity


class Randomize(Transform):
    """Apply the given transform randomly.

    Given a transform and the probability of applying it, randomly apply the transform.

    Args:
        transform: The transform to apply.
        probability: The probability to apply the transform, between 0 and 1.
    """

    def __init__(self, transform: callable, probability: float = 0.5):
        super().__init__()
        self.transform = transform
        self.identity = Identity()
        self.probability = probability

    def __call__(self, *args):
        if np.random.rand() < self.probability:
            return self.transform(*args)
        return self.identity(*args)
