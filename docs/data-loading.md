# Working with sign-language-data-loading

[`sign-language-data-loading`](https://github.com/ppoitier/sign-language-data-loading) (imported as
`sldl`) loads sign language datasets stored as WebDataset shards — poses, videos and temporal
annotations — and hands you back plain `numpy`/`pandas` samples. `sign-language-tools` picks up from
there: augmenting the poses, and visualizing or playing back the sample.

```bash
pip install sign-language-data-loading
```

## Loading a sample

```python
from sldl import SignLanguageDataset
from sldl.configs import LSFBContConfig

dataset = SignLanguageDataset.from_config(
    LSFBContConfig(
        root="path/to/lsfb-cont",
        split="testing",
        load_videos=False,
    )
)

sample = dataset[42]
sample["poses"]                 # dict[str, np.ndarray], e.g. {"upper_pose": (T, 23, 3), ...}
sample["annotations"]           # dict[str, pandas.DataFrame], per annotation track
```

For isolated-sign datasets, use `sldl.configs.LSFBIsolConfig` instead — each sample is then a single
sign with a `sample["label"]` instead of a DataFrame of annotations.

## Augmenting the poses

The pose arrays returned by `sldl` are plain `(T, L, C)` `numpy` arrays, so they plug directly into
[pose transforms](pose.md#transforms):

```python
from sign_language_tools.common.transforms import Compose
from sign_language_tools.pose.transforms import InterpolateMissing, TemporalRandomCrop, HorizontalFlip

augment = Compose([
    InterpolateMissing(),
    TemporalRandomCrop(size=64),
    HorizontalFlip(),
])

upper_pose = augment(sample["poses"]["upper_pose"])
```

## Visualizing the sample

Play back the pose keypoints with the [`VideoPlayer`](player.md), using the MediaPipe edge
definitions to draw the skeleton:

```python
from sign_language_tools.player import VideoPlayer
from sign_language_tools.pose.mediapipe.edges import UPPER_POSE_EDGES, HAND_EDGES

player = VideoPlayer()
player.attach_empty(width=800, height=600, name="pose data")
player.attach_poses(sample["poses"]["upper_pose"], UPPER_POSE_EDGES, parent_name="pose data")
player.attach_poses(
    sample["poses"]["left_hand"], HAND_EDGES, edge_color=(255, 0, 0), parent_name="pose data",
)
player.attach_poses(
    sample["poses"]["right_hand"], HAND_EDGES, edge_color=(0, 0, 255), parent_name="pose data",
)
player.play(speed=0.5)
```

If the dataset sample also carries a video (`load_videos=True`), attach it with
`player.attach_video_tensor(sample["video"], fps=...)` (or `attach_video_file` if you have the raw
file path) as the root component instead of `attach_empty`, and attach the poses as its children.

## Converting annotations for training

Continuous samples carry their annotations as a `pandas.DataFrame` of gloss segments
(`start_frame`, `end_frame`, `lemma`, ...). Convert them to a dense per-frame label vector with the
[annotations transforms](annotations.md#transforms), e.g. to train a frame-level classifier:

```python
import numpy as np
from sign_language_tools.annotations.transforms import SegmentsToFrameLabels

annotations = sample["annotations"]["both_hands"]
segments = annotations[["start_frame", "end_frame"]].to_numpy()

frame_labels = SegmentsToFrameLabels()(segments, vector_size=sample["n_frames"])
```

## Batching for training

`sldl.SignLanguageCollator` batches samples (padding pose sequences and building a mask), and the
[common transforms](common.md) `Compose`/`ApplyToAll` can be passed as the dataset's `transform` to
apply pose augmentation per-sample before batching:

```python
from torch.utils.data import DataLoader
from sldl import SignLanguageCollator

loader = DataLoader(dataset, batch_size=8, collate_fn=SignLanguageCollator())
batch = next(iter(loader))
batch["poses"]["upper_pose"].shape   # (batch, max_n_frames, ...)
batch["masks"].shape                 # (batch, max_n_frames)
```
