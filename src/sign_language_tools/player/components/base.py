from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from uuid import uuid4

import numpy as np


@dataclass()
class Component(ABC):
    """Base class for all visual components in the player.

    Every component knows how to render itself onto a frame via `to_frame`.
    Components form a tree: each component can have children that render
    on top of (into) the parent's frame.
    """

    name: str = field(default_factory=lambda: str(uuid4()))
    fps: float = 25.0
    speed: float = 1.0
    children: list['Component'] = field(default_factory=list)

    @abstractmethod
    def to_frame(self, t: float, parent_frame: np.ndarray | None = None) -> np.ndarray:
        """Render this component at time `t`.

        Args:
            t: Current playback time in seconds.
            parent_frame: If provided, render *into* this frame instead of
                creating a new one.  Leaf components that overlay a parent
                (skeletons, annotations, …) should draw onto `parent_frame`.

        Returns:
            The rendered frame (H×W×3 uint8 ndarray).
        """
        ...

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def frame_index(self, t: float) -> int:
        """Map a time in seconds to a 0-based frame index."""
        return max(0, round(t * self.fps))

    def add_child(self, child: 'Component') -> None:
        self.children.append(child)

    @staticmethod
    def _blank_frame(width: int, height: int, color: tuple[int, int, int] = (0, 0, 0)) -> np.ndarray:
        return np.full((height, width, 3), fill_value=[[color]], dtype="uint8")