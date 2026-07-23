import torch

from sign_language_tools.core.transform import Transform


class TemporalPad(Transform):
    """Pads a video along the temporal dimension by repeating its first or last frame.

    If the video already has at least `min_width` frames, it is returned unchanged.

    Args:
        min_width (int): The minimum number of frames the output video must have.
        location (str): Where to add the padding frames. Either `"start"` or `"end"`.

    Example:
        >>> import torch
        >>> from sign_language_tools.video.transforms import TemporalPad
        >>> video = torch.arange(3 * 3 * 2 * 2, dtype=torch.float32).reshape(3, 3, 2, 2)
        >>> pad = TemporalPad(min_width=5, location='end')
        >>> pad(video).shape
        torch.Size([5, 3, 2, 2])
    """

    def __init__(self, min_width: int, location: str = 'end'):
        super().__init__()
        self.min_width = min_width
        self.location = location

    def __call__(self, video: torch.Tensor) -> torch.Tensor:
        """Pads the video with repeated frames until it reaches `min_width` frames.

        Args:
            video (torch.Tensor): A video tensor with shape `(T, C, H, W)`.

        Returns:
            torch.Tensor: The padded video tensor with shape `(min_width, C, H, W)`,
                or the original video if it already has at least `min_width` frames.
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
