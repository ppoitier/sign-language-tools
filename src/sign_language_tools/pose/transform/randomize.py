import numpy as np


class Randomize:

    """Apply the given transform randomly.

    Given a transform and the probability of applying it, apply the transform with the given probability.

    Args:
        transform: The transformation to apply.
        probability: A value between 0 and 1 .

    Returns:
        landmarks: The resulting landmarks.
    """

    def __init__(self, transform: callable, probability: float = 0.5):
        self.transform = transform
        self.odds = probability

    def __call__(self, x: np.array) -> np.array:
        if np.random.rand() < self.odds:
            return self.transform(x)
        else:
            return x
