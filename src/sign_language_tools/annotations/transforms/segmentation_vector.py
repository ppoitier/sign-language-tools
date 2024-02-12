import math
from typing import Optional
from itertools import groupby

import numpy as np
import pandas as pd

from sign_language_tools.core.transform import Transform


class ToSegmentationVector(Transform):
    def __init__(
            self,
            vector_size: Optional[int] = None,
            background_label: int = 0,
            fill_label: int = 1,
            use_annotation_labels: bool = True,
            dtype='uint8',
    ):
        super().__init__()
        self.vector_size = vector_size
        self.background_label = background_label
        self.fill_label = fill_label
        self.use_annotation_labels = use_annotation_labels
        self.dtype = dtype

    def __call__(self, *args):
        if len(args) > 1:
            vec_size = len(args[0])
            annotations = args[1]
            if isinstance(annotations, pd.DataFrame):
                annotations = annotations.values
        else:
            annotations = args[0]
            if isinstance(annotations, pd.DataFrame):
                annotations = annotations.values
            if self.vector_size is None:
                vec_size = math.ceil(annotations[:, 1].max())
            else:
                vec_size = self.vector_size

        # segmentation = np.zeros(vec_size, dtype=self.dtype)
        segmentation = np.full(vec_size, fill_value=self.background_label, dtype=self.dtype)
        for annot in annotations:
            label = annot[2] if self.use_annotation_labels else self.fill_label
            segmentation[annot[0]:annot[1]] = label

        if len(args) > 1:
            return args[0], segmentation
        return segmentation


class FromSegmentationToAnnotations(Transform):
    def __init__(
            self,
            background_label: int = 0,
            fill_label: int = 1,
            use_annotation_labels: bool = True,
    ):
        super().__init__()
        self.background_label = background_label
        self.fill_label = fill_label
        self.use_annotation_labels = use_annotation_labels

    def __call__(self, segmentation: np.ndarray):
        segments = np.array([(k, len(list(g))) for k, g in groupby(segmentation)])
        segments_end = segments[:, 1].cumsum() + 1
        segments_start = np.zeros_like(segments_end)
        segments_start[1:] = segments_end[:-1] - 1
        segments = np.stack([segments_start, segments_end, segments[:, 0]], axis=1)
        segments = segments[segments[:, 2] != self.background_label]
        if not self.use_annotation_labels:
            segments[:, 2] = self.fill_label
        return segments
