# Common Transforms

`sign_language_tools.common.transforms` provides small, data-agnostic transform building blocks used
to combine and glue together the domain-specific transforms from [pose](pose.md), [video](video.md)
and [annotations](annotations.md).

## Example

```python
from sign_language_tools.common.transforms import Compose, Randomize, Identity
from sign_language_tools.pose.transforms import HorizontalFlip, GaussianNoise

augment = Compose([
    Randomize(HorizontalFlip(), probability=0.5),
    Randomize(GaussianNoise(), probability=0.3),
])

pose_sequence = augment(pose_sequence)
```

`Compose` also supports chaining transforms that take/return multiple positional arguments (e.g. a
pose sequence and its segments together) via `multi_input=True`.

## Reference

::: sign_language_tools.common.transforms.compose.Compose
::: sign_language_tools.common.transforms.concatenate.Concatenate
::: sign_language_tools.common.transforms.identity.Identity
::: sign_language_tools.common.transforms.map.MapTransform
::: sign_language_tools.common.transforms.map.ApplyToAll
::: sign_language_tools.common.transforms.randomize.Randomize
::: sign_language_tools.common.transforms.tuple.TransformTuple
::: sign_language_tools.common.transforms.replace_nan.ReplaceNaN
