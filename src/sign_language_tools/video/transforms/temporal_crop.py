import numpy as np
import torch

from sign_language_tools.core.transform import Transform


class TemporalCrop(Transform):
    def __init__(self, max_width: int, location: str = 'start'):
        super().__init__()
        self.max_width = max_width
        self.location = location

    def __call__(self, pose_seq: np.ndarray) -> np.ndarray:
        seq_len = pose_seq.shape[0]
        if seq_len <= self.max_width:
            return pose_seq
        if self.location == 'start':
            return pose_seq[:self.max_width]
        elif self.location == 'center':
            start_idx = (pose_seq.shape[0] - self.max_width) // 2
            return pose_seq[start_idx:start_idx + self.max_width]
        elif self.location == 'end':
            return pose_seq[-self.max_width:]
        else:
            raise ValueError(f"Unknown location: {self.location}. Please use 'start', 'center', or 'end'.")


class TemporalRandomCrop(Transform):
    def __init__(self, max_width: int):
        super().__init__()
        self.max_width = max_width

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        """
        Args:
            video (torch.Tensor): A video tensor with shape (T, C, H, W)
        """
        T = video.shape[0]
        if T <= self.max_width:
            return video

        start_index = torch.randint(0, T - self.max_width + 1, (1,)).item()
        return video[start_index : start_index + self.max_width]
