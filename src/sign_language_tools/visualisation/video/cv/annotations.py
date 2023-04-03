import cv2
import numpy as np
import pandas as pd

from .utils import draw_rect, draw_line, absolute_position


def draw_annot_rect(img, x_start, x_end):
    draw_rect(img, (x_start, 0.5), (x_end, 1.0), color=(51, 204, 51))


def draw_annot_text(img, text, x_start, y_text):
    draw_line(img, (x_start, 1), (x_start, y_text - 0.05), color=(51, 204, 51), thickness=1)
    cv2.putText(img, text, absolute_position(img, (x_start, y_text)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, 2)


def to_img_x(f: int, panel_start: int, panel_end: int, width: int):
    x = (f - panel_start) * width / (panel_end - panel_start)
    return min(width, max(0, round(x)))


def draw_annotation(img: np.ndarray, index: int, annot, panel_start: int, panel_end: int):
    width = img.shape[1]
    height = img.shape[0]

    x0 = to_img_x(annot['start'], panel_start, panel_end, width)
    x1 = to_img_x(annot['end'], panel_start, panel_end, width)

    color = (0, 255, 0)
    if 'color' in annot:
        r, g, b = annot[['r', 'g', 'b']]
        color = (b, g, r)

    img[-100:, x0:x1, :] = color

    text_x = x0 + 2
    text_y0 = height - (100 + (1 + index % 4) * 30)
    text_y1 = height - 100

    cv2.line(img, (text_x, text_y0), (text_x, text_y1), color=(0, 255, 0), thickness=1)
    cv2.putText(img, str(annot['label']), (text_x + 2, text_y0), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, 2)


def draw_annotations(img: np.ndarray, annots: pd.DataFrame, panel_start: int, panel_end: int):
    for index, annot in annots.iterrows():
        draw_annotation(img, index, annot, panel_start, panel_end)
