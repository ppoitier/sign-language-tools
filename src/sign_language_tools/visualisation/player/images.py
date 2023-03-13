import numpy as np
import cv2
import glob
import os
from .displayable import Displayable


class Images(Displayable):

    def __init__(self, dir_path: str, extension: str = 'png'):
        super(Images, self).__init__()

        self.dir_path = dir_path
        self.frames = [cv2.imread(filepath) for filepath in glob.glob(os.path.join(dir_path, f'*.{extension}'))]

        self.frame_count = len(self.frames)
        self.width = self.frames[0].shape[1]
        self.height = self.frames[0].shape[0]

        self.last_frame_nb = 0

    def get_img(self, frame_number: int) -> np.ndarray:
        frame_number = max(0, min(frame_number, self.frame_count-1))

        return self.frames[frame_number]
