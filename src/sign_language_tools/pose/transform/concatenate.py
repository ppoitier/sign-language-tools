from typing import Union, Sequence

import numpy as np

from sign_language_tools.core.transform import Transform


class Concatenate(Transform):
    """Concatenate multiple pose arrays along the landmark axis.

    Accepts either a dict mapping body part names to pose arrays
    (e.g., {"upper_body": ..., "left_hand": ..., "right_hand": ...})
    or a list of pose arrays. All arrays must have compatible shapes
    along every axis except the concatenation axis.

    Args:
        landmark_sets: Ordered list of keys to select and order entries
            from a dict input. Ignored when the input is a list.
            If None, keys are sorted alphabetically.
        landmark_axis: Axis along which to concatenate. Defaults to 1
            (the L dimension for arrays of shape (T, L, C)).
    """

    def __init__(
        self,
        landmark_sets: Sequence[str] = None,
        landmark_axis: int = 1,
    ):
        super().__init__()
        self.landmark_sets = landmark_sets
        self.landmark_axis = landmark_axis

    def __call__(
        self, poses: Union[dict[str, np.ndarray], list[np.ndarray]]
    ) -> np.ndarray:
        if isinstance(poses, dict):
            keys = self.landmark_sets or sorted(poses.keys())
            missing = set(keys) - poses.keys()
            if missing:
                raise KeyError(f"Missing landmark sets in input: {missing}")
            poses = [poses[key] for key in keys]

        if not poses:
            raise ValueError("Cannot concatenate an empty sequence of poses.")

        return np.concatenate(poses, axis=self.landmark_axis)
