import cv2
import numpy as np
import pandas as pd


def absolute_position(img, pos):
    w = int(img.shape[1])
    h = int(img.shape[0])
    return int(max(0, min(w, w * pos[0]))), int(max(0, min(h, h * pos[1])))


def compute_box(pos, box_size):
    x, y = pos
    x1 = int(max(0, x - box_size / 2))
    y1 = int(max(0, y - box_size / 2))
    x2 = int(x + box_size / 2)
    y2 = int(y + box_size / 2)
    return x1, y1, x2, y2


def extract_box_img(img, box, dim=(256, 256)):
    x1, y1, x2, y2 = box
    box_img = img[y1:y2, x1:x2]

    if box_img.shape[0] != box_img.shape[1]:
        box_img = squarify_image(box_img)

    return cv2.resize(box_img, dim, interpolation=cv2.INTER_AREA)


def squarify_image(img):
    height = img.shape[0]
    width = img.shape[1]

    if height > width:
        padding = ((0, 0), (0, height-width), (0, 0))
    else:
        padding = ((0, width-height), (0, 0), (0, 0))

    return np.pad(img, padding, mode='constant', constant_values=0)


def get_feature_box(img, rel_pos: (float, float), holistic_features: pd.DataFrame, shoulder_length_factor: float):
    abs_pos = absolute_position(img, rel_pos)

    right_shoulder = np.array(absolute_position(img, (
        holistic_features['RIGHT_SHOULDER_X'],
        holistic_features['RIGHT_SHOULDER_Y']
    )))
    left_shoulder = np.array(absolute_position(img, (
        holistic_features['LEFT_SHOULDER_X'],
        holistic_features['LEFT_SHOULDER_Y']
    )))

    base_length = max(50, np.sqrt(np.sum(np.square(right_shoulder - left_shoulder))))
    return compute_box(abs_pos, base_length * shoulder_length_factor)
