# Pose

The `pose` submodule covers the full lifecycle of pose landmarks: extracting them from video with
MediaPipe, describing their skeleton connectivity, transforming/augmenting them, and plotting them.

Across the submodule, a pose sequence is represented as a `numpy` array of shape `(T, L, C)`, where
`T` is the number of frames, `L` the number of landmarks, and `C` the number of coordinates per
landmark (usually 3: x, y, z).

## Extracting landmarks with MediaPipe

`sign_language_tools.pose.mediapipe.extraction` wraps MediaPipe's holistic landmarker to extract
pose, hands and face landmarks from a video file frame by frame.

```python
from sign_language_tools.pose.mediapipe.extraction import (
    load_holistic_landmarker,
    extract_poses_from_video_file,
)

landmarker = load_holistic_landmarker("holistic_landmarker.task")
poses = extract_poses_from_video_file("video.mp4", landmarker, show_progress=True)

poses["pose"].shape        # (T, 33, 3)
poses["left_hand"].shape   # (T, 21, 3)
poses["right_hand"].shape  # (T, 21, 3)
poses["face"].shape        # (T, 478, 3)
```

![Extracted landmarks (skeleton only, skeleton overlaid on the frame, and the original frame)](figures/example_sample.jpg)

Frames where a body part wasn't detected are filled with `NaN`; see
[`InterpolateMissing`][sign_language_tools.pose.transforms.interpolate.InterpolateMissing] to fill
them in.

## Skeleton connectivity

`sign_language_tools.pose.mediapipe.edges` and `sign_language_tools.pose.mediapipe.vertices` provide
the connectivity (edges) and landmark groupings (vertices) for the MediaPipe skeletons, and
`sign_language_tools.pose.openpose.edges` provides the equivalent for OpenPose. These are typically
used to draw skeletons (see [Visualization](#visualization) and the [Video Player](player.md)) or to
build graph structures over the landmarks.

```python
from sign_language_tools.pose.mediapipe.edges import POSE_EDGES, HAND_EDGES, FACE_EDGES
from sign_language_tools.pose.mediapipe.vertices import LIPS_VERTICES
```

## Transforms

`sign_language_tools.pose.transforms` provides data-augmentation and preprocessing transforms for
pose sequences, compatible with PyTorch/TensorFlow data pipelines. They can be composed with
[`Compose`][sign_language_tools.common.transforms.compose.Compose] from the [common](common.md)
submodule.

```python
import numpy as np
from sign_language_tools.pose.transforms import (
    Concatenate, TemporalRandomCrop, HorizontalFlip, Split, InterpolateMissing,
)

landmarks = {
    "pose": np.load("examples/ressources/pose.npy"),
    "left_hand": np.load("examples/ressources/left_hand.npy"),
    "right_hand": np.load("examples/ressources/right_hand.npy"),
}

pipeline = [
    InterpolateMissing(),
    Concatenate(["pose", "right_hand", "left_hand"]),
    TemporalRandomCrop(size=60),
    HorizontalFlip(),
    Split({"pose": 33, "right_hand": 21, "left_hand": 21}),
]

x = landmarks
for transform in pipeline:
    x = transform(x)
```

Transforms operating on a single array (e.g. `TemporalRandomCrop`, `HorizontalFlip`,
`InterpolateMissing`) take and return a `(T, L, C)` array. `Concatenate` and `Split` bridge between
a `dict[str, np.ndarray]` of landmark groups (e.g. `{"pose": ..., "left_hand": ...}`) and a single
concatenated `(T, L, C)` array, so that group-agnostic transforms can be applied uniformly.

## Visualization

`sign_language_tools.pose.visualization` provides matplotlib plotting for landmarks, one frame or a
whole sequence at a time. For interactive playback synced to a video, see the [Video Player](player.md)
instead.

```python
from sign_language_tools.pose.visualization import plot_landmarks, plot_landmarks_sequence
from sign_language_tools.pose.mediapipe.edges import POSE_EDGES, HAND_EDGES

# A single frame
plot_landmarks(landmarks["pose"][0], connections=POSE_EDGES)

# A short sequence, one subplot per frame
fig = plot_landmarks_sequence(
    {"pose": landmarks["pose"][:5], "right_hand": landmarks["right_hand"][:5]},
    {"pose": POSE_EDGES, "right_hand": HAND_EDGES},
)
```

## Reference

### Extraction

::: sign_language_tools.pose.mediapipe.extraction.load_holistic_landmarker
::: sign_language_tools.pose.mediapipe.extraction.extract_poses_from_video_file

### Edges and vertices

::: sign_language_tools.pose.mediapipe.edges
    options:
        show_source: false
        members: false
::: sign_language_tools.pose.mediapipe.vertices
    options:
        show_source: false
        members: false
::: sign_language_tools.pose.openpose.edges
    options:
        show_source: false
        members: false

### Transforms

::: sign_language_tools.pose.transforms.center.CenterOnLandmarks
::: sign_language_tools.pose.transforms.clip.Clip
::: sign_language_tools.pose.transforms.concatenate.Concatenate
::: sign_language_tools.pose.transforms.drop_coordinates.DropCoordinates
::: sign_language_tools.pose.transforms.drop_frames.DropRandomFrames
::: sign_language_tools.pose.transforms.edge_normalize.NormalizeByReferenceEdge
::: sign_language_tools.pose.transforms.filter.FilterEmpty
::: sign_language_tools.pose.transforms.filter.FilterLandmarks
::: sign_language_tools.pose.transforms.flatten.Flatten
::: sign_language_tools.pose.transforms.flatten.Unflatten
::: sign_language_tools.pose.transforms.flip.HorizontalFlip
::: sign_language_tools.pose.transforms.interpolate.InterpolateMissing
::: sign_language_tools.pose.transforms.noise.GaussianNoise
::: sign_language_tools.pose.transforms.normalize.MinMaxNormalization
::: sign_language_tools.pose.transforms.normalize.Standardization
::: sign_language_tools.pose.transforms.normalize.FixedResolutionNormalization
::: sign_language_tools.pose.transforms.optical_flow.ToOpticalFlow
::: sign_language_tools.pose.transforms.padding.Padding
::: sign_language_tools.pose.transforms.resample.Resample
::: sign_language_tools.pose.transforms.resample.RandomResample
::: sign_language_tools.pose.transforms.rotate_2d.Rotation2D
::: sign_language_tools.pose.transforms.rotate_2d.RandomRotation2D
::: sign_language_tools.pose.transforms.rotate_2d.MakeReferenceEdgeHorizontal
::: sign_language_tools.pose.transforms.scale.Scale
::: sign_language_tools.pose.transforms.scale.RandomScale
::: sign_language_tools.pose.transforms.smoothing.SavitzkyGolayFiltering
::: sign_language_tools.pose.transforms.split.Split
::: sign_language_tools.pose.transforms.temporal_crop.TemporalCrop
::: sign_language_tools.pose.transforms.temporal_crop.TemporalRandomCrop
::: sign_language_tools.pose.transforms.temporal_scale.TemporalScale
::: sign_language_tools.pose.transforms.temporal_scale.RandomTemporalScale
::: sign_language_tools.pose.transforms.to_img.ToRGBImage
::: sign_language_tools.pose.transforms.translation.Translation
::: sign_language_tools.pose.transforms.translation.RandomTranslation

### Visualization

::: sign_language_tools.pose.visualization.landmarks.plot_landmarks
::: sign_language_tools.pose.visualization.landmarks.plot_landmarks_sequence
