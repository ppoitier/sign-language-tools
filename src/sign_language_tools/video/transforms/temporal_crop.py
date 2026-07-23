import torch

from sign_language_tools.core.transform import Transform


class TemporalCrop(Transform):
    """Crops a video along the temporal dimension to a fixed number of frames.

    If the video already has at most `max_width` frames, it is returned unchanged.

    Args:
        max_width (int): The maximum number of frames the output video can have.
        location (str): Where to take the frames from. Either `"start"`, `"center"`, or `"end"`.

    Example:
        >>> import torch
        >>> from sign_language_tools.video.transforms import TemporalCrop
        >>> video = torch.arange(5 * 3 * 2 * 2, dtype=torch.float32).reshape(5, 3, 2, 2)
        >>> crop = TemporalCrop(max_width=3, location='start')
        >>> crop(video).shape
        torch.Size([3, 3, 2, 2])
    """

    def __init__(self, max_width: int, location: str = 'start'):
        super().__init__()
        self.max_width = max_width
        self.location = location

    def __call__(self, video: torch.Tensor) -> torch.Tensor:
        """Crops the video down to `max_width` frames.

        Args:
            video (torch.Tensor): A video tensor with shape `(T, C, H, W)`.

        Returns:
            torch.Tensor: The cropped video tensor with shape `(max_width, C, H, W)`,
                or the original video if it already has at most `max_width` frames.
        """
        seq_len = video.shape[0]
        if seq_len <= self.max_width:
            return video
        if self.location == 'start':
            return video[:self.max_width]
        elif self.location == 'center':
            start_idx = (video.shape[0] - self.max_width) // 2
            return video[start_idx:start_idx + self.max_width]
        elif self.location == 'end':
            return video[-self.max_width:]
        else:
            raise ValueError(f"Unknown location: {self.location}. Please use 'start', 'center', or 'end'.")


class TemporalRandomCrop(Transform):
    """Crops a video along the temporal dimension to a fixed number of frames at a random offset.

    If the video already has at most `max_width` frames, it is returned unchanged.
    Otherwise, a window of `max_width` consecutive frames is selected starting at a
    uniformly random offset.

    Args:
        max_width (int): The maximum number of frames the output video can have.

    Example:
        >>> import torch
        >>> from sign_language_tools.video.transforms import TemporalRandomCrop
        >>> video = torch.arange(5 * 3 * 2 * 2, dtype=torch.float32).reshape(5, 3, 2, 2)
        >>> crop = TemporalRandomCrop(max_width=3)
        >>> crop(video).shape
        torch.Size([3, 3, 2, 2])
    """

    def __init__(self, max_width: int):
        super().__init__()
        self.max_width = max_width

    def __call__(self, video: torch.Tensor) -> torch.Tensor:
        """Crops the video down to `max_width` consecutive frames starting at a random offset.

        Args:
            video (torch.Tensor): A video tensor with shape `(T, C, H, W)`.

        Returns:
            torch.Tensor: The cropped video tensor with shape `(max_width, C, H, W)`,
                or the original video if it already has at most `max_width` frames.
        """
        T = video.shape[0]
        if T <= self.max_width:
            return video

        start_index = torch.randint(0, T - self.max_width + 1, (1,)).item()
        return video[start_index : start_index + self.max_width]
