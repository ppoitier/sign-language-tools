from typing import Union, Sequence

import numpy as np

from sign_language_tools.core.transform import Transform


class Concatenate(Transform):
    """Concatenates multiple pose sequences along the landmark axis.

    Accepts either a dict mapping body part names to pose sequences
    (e.g. `{"upper_body": ..., "left_hand": ..., "right_hand": ...}`) or a
    list of pose sequences. All pose sequences must have compatible shapes
    along every axis except the concatenation axis.

    Args:
        landmark_sets: Ordered list of keys to select and order entries
            from a dict input. Ignored when the input is a list. If None,
            keys are sorted alphabetically.
        landmark_axis: Axis along which to concatenate. Defaults to `1`
            (the `L` dimension for pose sequences of shape `(T, L, C)`).

    Example:
        >>> import numpy as np
        >>> from sign_language_tools.pose.transform import Concatenate
        >>> upper_body = np.random.rand(10, 8, 2)  # (T, L, C)
        >>> left_hand = np.random.rand(10, 21, 2)  # (T, L, C)
        >>> transform = Concatenate(landmark_sets=["upper_body", "left_hand"])
        >>> pose_sequence = transform({"upper_body": upper_body, "left_hand": left_hand})
        >>> pose_sequence.shape
        (10, 29, 2)
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
        self, pose_sequences: Union[dict[str, np.ndarray], list[np.ndarray]]
    ) -> np.ndarray:
        """Concatenates the pose sequences.

        Args:
            pose_sequences: Either a dict mapping body part names to pose
                sequences, or a list of pose sequences, each of shape
                `(T, L, C)` (with a possibly different `L` per entry).

        Returns:
            The concatenated pose sequence, of shape `(T, sum(L), C)`.
        """
        if isinstance(pose_sequences, dict):
            keys = self.landmark_sets or sorted(pose_sequences.keys())
            missing = set(keys) - pose_sequences.keys()
            if missing:
                raise KeyError(f"Missing landmark sets in input: {missing}")
            pose_sequences = [pose_sequences[key] for key in keys]

        if not pose_sequences:
            raise ValueError("Cannot concatenate an empty sequence of poses.")

        return np.concatenate(pose_sequences, axis=self.landmark_axis)
