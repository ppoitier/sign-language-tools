import torch

from sign_language_tools.core.transform import Transform


class TemporalPad(Transform):
    def __init__(self, min_width: int, location='end'):
        super().__init__()
        self.min_width = min_width
        self.location = location

    def __call__(self, video: torch.Tensor) -> torch.Tensor:
        """
        Args:
            video (torch.Tensor): A video tensor with shape (T, C, H, W)
        """
        T = video.shape[0]
        if T >= self.min_width:
            return video

        num_padding = self.min_width - T
        if self.location == 'start':
            first_frame = video[:1, :, :, :]
            padding = first_frame.repeat(num_padding, 1, 1, 1)
            return torch.cat([padding, video], dim=0)
        elif self.location == 'end':
            last_frame = video[-1:, :, :, :]
            padding = last_frame.repeat(num_padding, 1, 1, 1)
            return torch.cat([video, padding], dim=0)
        else:
            raise ValueError(f"Unknown padding location: {self.location}")
