# Video

The `video` submodule handles video decoding and tensor-level transforms. For interactive playback
and visualization, see the [Video Player](player.md) instead.

## Decoding

`sign_language_tools.video.decoding` iterates over the frames of a video file using
[`vidgear`](https://abhitronix.github.io/vidgear/), yielding each frame together with its estimated
timestamp. It's the same decoder used internally by
[`extract_poses_from_video_file`][sign_language_tools.pose.mediapipe.extraction.extract_poses_from_video_file].

```python
from sign_language_tools.video.decoding import iterate_video_frames_using_vidgear

for timestamp_ms, frame in iterate_video_frames_using_vidgear("video.mp4", show_progress=True):
    ...  # frame is an RGB array of shape (H, W, 3)
```

## Transforms

`sign_language_tools.video.transforms` provides PyTorch-tensor transforms for cropping and padding
video clips along the temporal dimension, mirroring the pose
[`TemporalCrop`/`TemporalRandomCrop`](pose.md#transforms) transforms but for `(T, C, H, W)` video
tensors.

```python
import torch
from sign_language_tools.video.transforms import TemporalCrop, TemporalPad

clip = torch.rand(120, 3, 224, 224)  # (T, C, H, W)

clip = TemporalCrop(max_width=64)(clip)
clip = TemporalPad(min_width=96)(clip)
```

## Reference

::: sign_language_tools.video.decoding.iterate_video_frames_using_vidgear
::: sign_language_tools.video.transforms.temporal_crop.TemporalCrop
::: sign_language_tools.video.transforms.temporal_crop.TemporalRandomCrop
::: sign_language_tools.video.transforms.temporal_padding.TemporalPad
