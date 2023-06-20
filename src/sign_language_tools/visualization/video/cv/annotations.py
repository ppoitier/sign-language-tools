import cv2
import numpy as np
import pandas as pd


def to_img_x(f: int, panel_start: int, panel_end: int, width: int):
    x = (f - panel_start) * width / (panel_end - panel_start)
    return min(width, max(0, round(x)))


def draw_segment(img: np.ndarray, index: int, segment, panel_start: int, panel_end: int):
    width = img.shape[1]
    height = img.shape[0]

    x0 = to_img_x(segment[0], panel_start, panel_end, width)
    x1 = to_img_x(segment[1], panel_start, panel_end, width)

    color = (0, 255, 0)
    if 'color' in segment:
        r, g, b = segment[['r', 'g', 'b']]
        color = (b, g, r)

    img[-100:, x0:x1, :] = color

    if segment.shape[0] > 2:
        text_x = x0 + 2
        text_y0 = height - (100 + (1 + index % 4) * 30)
        text_y1 = height - 100

        cv2.line(img, (text_x, text_y0), (text_x, text_y1), color=(0, 255, 0), thickness=1)
        cv2.putText(img, str(segment[2]), (text_x + 2, text_y0), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, 2)


def draw_segments(img: np.ndarray, segments: pd.DataFrame, panel_start: int, panel_end: int):
    for index, segment in segments.iterrows():
        index: int
        draw_segment(img, index, segment, panel_start, panel_end)
