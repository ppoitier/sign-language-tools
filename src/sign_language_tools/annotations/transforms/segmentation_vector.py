import math
from typing import Optional
from itertools import groupby

import numpy as np
import pandas as pd

from sign_language_tools.core.transform import Transform


class SegmentsToSegmentationVector(Transform):
    def __init__(
            self,
            vector_size: Optional[int] = None,
            background_label: int = -1,
            fill_label: int = 1,
            use_annotation_labels: bool = True,
            dtype='int32',
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

        segmentation = np.full(vec_size, fill_value=self.background_label, dtype=self.dtype)
        for annot in annotations:
            label = annot[2] if self.use_annotation_labels else self.fill_label
            segmentation[annot[0]:annot[1]+1] = label
        if len(args) > 1:
            return args[0], segmentation
        return segmentation


class SegmentationVectorToSegments(Transform):
    def __init__(
            self,
            background_classes: tuple[int, ...] = (0,),
            use_annotation_labels: bool = True,
    ):
        super().__init__()
        self.background_classes = background_classes
        self.use_annotation_labels = use_annotation_labels

    def __call__(self, segmentation: np.ndarray):
        # Find where values change (including the first and last position)
        change_indices = np.nonzero(np.diff(segmentation))[0] + 1
        change_indices = np.r_[0, change_indices, len(segmentation)]
        lengths = np.diff(change_indices)
        values = segmentation[change_indices[:-1]]

        segments_end = np.cumsum(lengths) - 1
        segments_start = np.zeros_like(segments_end)
        segments_start[1:] = segments_end[:-1] + 1
        segments = np.stack([segments_start, segments_end, values], axis=1)

        segments = segments[np.isin(segments[:, 2], self.background_classes, invert=True)]
        if not self.use_annotation_labels:
            return segments[:, :2]
        return segments
