
import cv2
import numpy as np

from sign_language_tools.player.components.base import Component


def render_tree(
    component: Component,
    t: float,
    parent_frame: np.ndarray | None = None,
) -> np.ndarray | None:
    """Recursively render a component and all its children.

    Args:
        component: Root of the (sub-)tree to render.
        t: Current playback time in seconds.
        parent_frame: Frame to render into, or *None* for root components.

    Returns:
        The composed frame, or *None* if a video stream has ended.
    """
    frame = component.to_frame(t, parent_frame)

    # A VideoComponent returning a black frame when the stream ended is the
    # "end" signal.  A more explicit mechanism could be added later.
    if frame is None:
        return None

    for child in component.children:
        frame = render_tree(child, t, parent_frame=frame)
        if frame is None:
            return None

    return frame


def display_frame(name: str, frame: np.ndarray) -> None:
    """Show a frame in an OpenCV window."""
    cv2.imshow(name, frame)