# Sign Language Tools

**Sign Language Tools** is a Python library that provides the building blocks needed to work with
sign language datasets made of videos, pose landmarks (from [MediaPipe](https://developers.google.com/mediapipe)
or [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose)) and their temporal
annotations (glosses, lemmas, sign boundaries, etc.).

It was originally developed by the [LSFB](https://lsfb.info.unamur.be/) team at the University of
Namur to build the machine-learning pipeline behind the LSFB dataset, and was later generalized to
work with other sign language datasets.

The library is designed to pair with
[`sign-language-data-loading`](https://github.com/ppoitier/sign-language-data-loading), which loads
poses, videos and annotations from a dataset on disk. `sign-language-data-loading` gets you a sample
into memory; `sign-language-tools` is what you use to extract, transform, visualize and play it back.
See [Working with sign-language-data-loading](data-loading.md) for a full example combining the two.

## What's included

- **[Pose](pose.md)** — Extract MediaPipe holistic landmarks from video, connectivity/edge
  definitions for MediaPipe and OpenPose skeletons, data-augmentation and preprocessing transforms
  (interpolation, resampling, smoothing, normalization, cropping, ...), and matplotlib plotting
  helpers for landmarks.
- **[Video](video.md)** — Decode video files frame by frame, and PyTorch-compatible transforms for
  video tensors (temporal crop, temporal padding).
- **[Video Player](player.md)** — An OpenCV-based, modular player to visualize a video alongside its
  pose landmarks, annotation segments, time series or heatmaps, all synchronized on a single
  playback clock.
- **[Annotations](annotations.md)** — Transforms to manipulate segment-based annotations (merging,
  scaling, converting to frame labels or BIO tags, ...), plus IoU/NMS utilities and a timeline
  plotting helper.
- **[Common Transforms](common.md)** — Small, data-agnostic transform building blocks (`Compose`,
  `Concatenate`, `Randomize`, ...) used to build transform pipelines across the other submodules.

## Project status

This project is under active development. Contributions and issues are welcome on
[GitHub](https://github.com/ppoitier/sign-language-tools).
