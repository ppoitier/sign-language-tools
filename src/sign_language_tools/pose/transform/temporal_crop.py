import numpy as np

from sign_language_tools.core.transform import Transform


class TemporalCrop(Transform):
    def __init__(self, size: int, location: str = 'start'):
        super().__init__()
        self.size = size
        self.location = location

    def __call__(self, pose_seq: np.ndarray) -> np.ndarray:
        seq_len = pose_seq.shape[0]
        if seq_len <= self.size:
            return pose_seq
        if self.location == 'start':
            return pose_seq[:self.size]
        elif self.location == 'center':
            start_idx = (pose_seq.shape[0] - self.size) // 2
            return pose_seq[start_idx:start_idx + self.size]
        elif self.location == 'end':
            return pose_seq[-self.size:]
        else:
            raise ValueError(f"Unknown location: {self.location}. Please use 'start', 'center', or 'end'.")


class TemporalRandomCrop(Transform):
    def __init__(self, size: int):
        super().__init__()
        self.size = size

    def __call__(self, landmarks: np.ndarray):
        T = landmarks.shape[0]
        if T <= self.size:
            return landmarks
        t = np.random.randint(low=0, high=T - self.size)
        return landmarks[t:t + self.size]
