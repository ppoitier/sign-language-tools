# Sign Language Tools 👋

This repository provides several tools aiming at facilitating the manipulation of sign language datasets. It was first developped by the authors of the [LSFB dataset](https://lsfb.info.unamur.be/) and then extended to other dataset using videos, gloss annotation and pose landmarks to ease the setup of our ML pipeline.

The project is made of two modules:

- **visualization**: this module provides tools to visualise sign language dataset along with their annotations and pose landmarks. It provides a convenient interface to display or save videos and frames.
- **pose**: This module provides functions and datatype to manipulate poses extracted via [Mediapipe](https://developers.google.com/mediapipe) or [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose). Transforms functions are implemented to ease the pre-processing and data augmentation of pose data.

This module is used internally by the researcher from the University of Namur but is made open to everyone who want to use it and contributes under the MIT License.
