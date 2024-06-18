import numpy as np

from sign_language_tools.core.transform import Transform


class Split(Transform):
    def __init__(self, groups: dict[str, int]):
        super().__init__()
        self.groups = groups

    def __call__(self, landmarks: np.ndarray):
        lm_counter = 0
        instance_groups = {}
        for key, n_landmarks in self.groups.items():
            instance_groups[key] = landmarks[:, lm_counter:lm_counter+n_landmarks]
            lm_counter += n_landmarks
        return instance_groups
