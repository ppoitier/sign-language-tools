from typing import List, Literal, Tuple

import numpy as np

from sign_language_tools.features.landmarks.indices import FACE_INDICES, HAND_INDICES, POSE_INDICES, LIPS_INDICES

Segment = Literal['face', 'lips', 'hand', 'pose']


def split_into_segments(landmarks: np.ndarray, segment_names: List[str]) -> Tuple[np.ndarray, ...]:
    segments = []
    offset = 0

    for name in segment_names:
        if name == 'face':
            indices = FACE_INDICES
        elif name == 'hand':
            indices = HAND_INDICES
        elif name == 'pose':
            indices = POSE_INDICES
        elif name == 'lips':
            indices = LIPS_INDICES
        else:
            raise ValueError(f'Unknown segment [{name}].')

        indices = np.array(indices) + offset
        offset += len(indices)

        segments.append(landmarks[:, indices])

    return tuple(segments)
