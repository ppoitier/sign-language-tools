# Annotations

The `annotations` submodule manipulates segment-based annotations: gloss/lemma boundaries, sign
activity, or any other labeling of a video's timeline.

A set of segments is represented as a `numpy` array of shape `(M, 2)` (`[start, end]`) or `(M, 3)`
(`[start, end, label]`), with an **inclusive end**: `[start, end]` covers frames `start, ..., end`.
This is the same convention used by [`VideoPlayer.attach_segments`](player.md#reference).

## Transforms

`sign_language_tools.annotations.transforms` converts and manipulates segments: merging adjacent or
overlapping spans, rendering them as a dense per-frame label vector (and back), scaling/moving them
for data augmentation, or building BIO tags for sequence labeling.

```python
import numpy as np
from sign_language_tools.annotations.transforms import (
    MergeSegments, SegmentsToFrameLabels, FrameLabelsToSegments,
)

segments = np.array([
    [2, 5, 1],
    [6, 12, 2],
    [14, 17, 0],
    [18, 25, 1],
])  # (M, 3): [start, end, label]

segments = MergeSegments()(segments)

frame_labels = SegmentsToFrameLabels()(segments)   # (T,) dense label vector
segments_back = FrameLabelsToSegments()(frame_labels)
```

## Utilities

`sign_language_tools.annotations.utils` provides temporal IoU (intersection-over-union) and
non-maximum suppression, useful to evaluate or post-process segment predictions.

```python
from sign_language_tools.annotations.utils.iou import pairwise_temporal_intersection_over_union
from sign_language_tools.annotations.utils.nms import non_maximum_suppression

tiou_map = pairwise_temporal_intersection_over_union(target_segments, predicted_segments)
kept = non_maximum_suppression(predicted_segments, iou_threshold=0.5)
```

## Visualization

`sign_language_tools.annotations.visualization` plots segments as colored spans on a matplotlib
timeline.

```python
import numpy as np
from sign_language_tools.annotations.visualization import plot_segments_on_timeline

segments = np.array([[0, 10], [15, 25]])
plot_segments_on_timeline(segments, labels=["bonjour", "merci"])
```

## Reference

### Transforms

::: sign_language_tools.annotations.transforms.scale.ScaleSegments
::: sign_language_tools.annotations.transforms.scale.RandomRelativeScaleSegments
::: sign_language_tools.annotations.transforms.boundaries.SegmentsToBoundaries
::: sign_language_tools.annotations.transforms.bio_tags.BioTags
::: sign_language_tools.annotations.transforms.offset.SegmentsToBoundaryOffsets
::: sign_language_tools.annotations.transforms.silence.CloseShortSilences
::: sign_language_tools.annotations.transforms.frame_labels.SegmentsToFrameLabels
::: sign_language_tools.annotations.transforms.frame_labels.FrameLabelsToSegments
::: sign_language_tools.annotations.transforms.merge.MergeSegmentsOnTransition
::: sign_language_tools.annotations.transforms.merge.MergeSegments
::: sign_language_tools.annotations.transforms.move.MoveSegments
::: sign_language_tools.annotations.transforms.move.RandomRelativeMoveSegments
::: sign_language_tools.annotations.transforms.overlapping.RemoveOverlapping
::: sign_language_tools.annotations.transforms.fill_between.FillBetween

### Utilities

::: sign_language_tools.annotations.utils.iou.temporal_intersection_over_union
::: sign_language_tools.annotations.utils.iou.pairwise_temporal_intersection_over_union
::: sign_language_tools.annotations.utils.iou.get_segment_positive_negative_count
::: sign_language_tools.annotations.utils.iou.calculate_segment_prediction_metrics
::: sign_language_tools.annotations.utils.nms.non_maximum_suppression
::: sign_language_tools.annotations.utils.nms.soft_nms

### Visualization

::: sign_language_tools.annotations.visualization.timeline.plot_segments_on_timeline
