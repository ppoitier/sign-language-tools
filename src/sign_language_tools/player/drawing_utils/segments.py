import cv2
import numpy as np

from sign_language_tools.player.drawing_utils.utils import to_frame_coords


def draw_timeline_details(
    frame: np.ndarray,
    t: float,
    frame_lims: np.ndarray,
    t_lims: np.ndarray,
    y_timeline: float,
    ticks_color: tuple[int, int, int],
):
    y_min, y_max = frame_lims[1]
    key_pos = np.array(
        [
            [t_lims[0, 0], y_timeline],
            [t_lims[0, 1], y_timeline],
            [t, 0.5],
            [t, y_timeline],
            [t - 1, 0.5],
            [t - 1, y_timeline],
            [t + 1, 0.5],
            [t + 1, y_timeline],
        ]
    )
    key_pos = to_frame_coords(key_pos, t_lims, frame_lims)
    cv2.line(frame, key_pos[0], key_pos[1], ticks_color, 3)
    cv2.line(frame, key_pos[2], key_pos[3], ticks_color, 2)
    cv2.line(frame, key_pos[4], key_pos[5], ticks_color, 2)
    cv2.line(frame, key_pos[6], key_pos[7], ticks_color, 2)
    cv2.putText(
        frame,
        "t",
        (key_pos[3, 0] - 5, y_max),
        cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.5,
        color=ticks_color,
        thickness=1,
        lineType=cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        "t-1s",
        (key_pos[5, 0] - 10, y_max),
        cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.5,
        color=ticks_color,
        thickness=1,
        lineType=cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        "t+1",
        (key_pos[7, 0] - 10, y_max),
        cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.5,
        color=ticks_color,
        thickness=1,
        lineType=cv2.LINE_AA,
    )


def draw_segments(
    frame: np.ndarray,
    sorted_segments: np.ndarray,
    t: float,
    t_lims: np.ndarray,
    frame_lims: np.ndarray,
    labels: list[str] | None = None,
    ticks_color: tuple[int, int, int] = (255, 255, 255),
    background_color: tuple[int, int, int] | None = None,
    segment_color: tuple[int, int, int] = (0, 200, 0),
    text_color: tuple[int, int, int] = (255, 255, 255),
    n_text_lines: int = 4,
    filled: bool = False,
):
    t_lims = t_lims.copy()
    t_lims[0] += t
    y_segment_top = 0.1
    y_timeline = 0.75

    if background_color is not None:
        cv2.rectangle(frame, frame_lims[:, 0], frame_lims[:, 1], background_color, -1)

    segment_2d_rects = np.repeat(sorted_segments[:, :, None], [2], axis=-1)
    segment_2d_rects[:, 0, 1] = y_segment_top
    segment_2d_rects[:, 1, 1] = y_timeline
    segment_2d_rects = to_frame_coords(segment_2d_rects, t_lims, frame_lims)

    x_min, x_max = frame_lims[0]
    for index, ((x_start, y_start), (x_end, y_end)) in enumerate(segment_2d_rects):
        if x_end >= x_min:
            x_start = max(x_start, x_min)
            x_end = min(x_end, x_max)
            cv2.rectangle(
                frame,
                (x_start, y_start),
                (x_end, y_end),
                segment_color,
                -1 if filled else 1,
            )
            if labels is not None:
                label = labels[index]
                label_x = x_start + (x_end - x_start) // 2 - 20
                label_level = 1 + (index % n_text_lines)
                label_y = y_start + label_level * 10
                cv2.putText(
                    frame,
                    label,
                    (label_x, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.4,
                    color=text_color,
                    thickness=1,
                    lineType=cv2.LINE_AA,
                )

        if x_start >= x_max:
            break
    draw_timeline_details(frame, t, frame_lims, t_lims, y_timeline, ticks_color)
    return frame


if __name__ == "__main__":
    h, w, c = 600, 800, 3

    t = np.linspace(0, 10, 300, dtype="float32")

    for current_t in t:
        frame = np.zeros((h, w, c), dtype=np.uint8)
        frame = draw_segments(
            frame=frame,
            sorted_segments=np.array(
                [
                    [0, 2],
                    [1, 2],
                    [3, 5],
                ],
                dtype="float32",
            ),
            labels=["sign 1", "sign 2", "sign 3"],
            t=current_t,
            t_lims=np.array([[-4, 4], [0, 1]], dtype="float32"),
            frame_lims=np.array([[100, 700], [400, 550]]),
            background_color=(0, 0, 100),
        )
        cv2.imshow("frame", frame)
        cv2.waitKey(20)
