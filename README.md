# Sign Language Tools

Python library containing various tools for sign language processing.

This project is in an early stage of development.
Do not hesitate to contribute!

📖 Documentation: [ppoitier.github.io/sign-language-tools](https://ppoitier.github.io/sign-language-tools/)

![Example of the video player showing a video with a hand skeleton overlay, alongside a per-hand annotation timeline](docs/figures/video_player.gif)

# Why?

Why not? It is useful to us, therefore it may be useful to you?

# What's included?

* **Pose** — Extract MediaPipe holistic landmarks from video, connectivity/edge definitions for
  MediaPipe and OpenPose skeletons, data-augmentation and preprocessing transforms (interpolation,
  resampling, smoothing, normalization, cropping, ...), and matplotlib plotting for landmarks.
* **Video** — Decode video files frame by frame, and PyTorch-compatible transforms for video
  tensors (temporal crop, temporal padding).
* **Video Player** — A modular, OpenCV-based player to visualize a video alongside its pose
  landmarks, annotation segments, time series or heatmaps.
* **Annotations** — Transforms to manipulate segment-based annotations (merging, scaling,
  converting to frame labels or BIO tags, ...), plus IoU/NMS utilities and a timeline plot.
* **Common transforms** — Small, data-agnostic building blocks (`Compose`, `Concatenate`,
  `Randomize`, ...) used to build transform pipelines across the other submodules.

This library pairs with [`sign-language-data-loading`](https://github.com/ppoitier/sign-language-data-loading),
which loads poses, videos and annotations from a dataset on disk — see the
[docs](https://ppoitier.github.io/sign-language-tools/data-loading/) for a full example combining
the two.

# Installation

```bash
pip install sign-language-tools
```

See the [installation guide](https://ppoitier.github.io/sign-language-tools/installation/) for
details and optional dependencies.

# What's next?

* More data augmentation and plotting!
* More tools for annotations.
* Tools for image processing.
* To be discussed...
