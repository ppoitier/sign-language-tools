import numpy as np
import pandas as pd
from .displayable import Displayable
from .cv.annotations import draw_annotations


def get_annots_in_range(annots: pd.DataFrame, full_range: tuple[int, int]) -> pd.DataFrame:
    return annots.loc[
        ((annots['start'] >= full_range[0]) & (annots['start'] <= full_range[1])) |
        ((annots['end'] >= full_range[0]) & (annots['end'] <= full_range[1])) |
        ((annots['start'] <= full_range[0]) & (annots['end'] >= full_range[1]))
    ]


class Annotations(Displayable):

    def __init__(
            self,
            annots: pd.DataFrame,
            name: str,
            unit: str,
            resolution: tuple[int, int] = (512, 256),
            time_range: int = 3000,
            fps: int = 50,
    ):
        self.name = name

        self.annots = annots
        self.resolution = resolution
        self.offset = resolution[0]//2
        self.time_range = time_range

        self.frame_duration = round(1000/fps)
        self.frame_offset = round(time_range/self.frame_duration)
        self.frame_px = self.offset/self.frame_offset

        self.panel = None
        self.panel_nb = 0

        self.unit = unit

        if self.unit == 'ms':
            self.annots.loc[:, ['start', 'end']] //= self.frame_duration

        self._load_panel(0)

    def get_img(self, frame_number: int) -> np.ndarray:
        panel_nb = frame_number // self.frame_offset
        if panel_nb != self.panel_nb:
            self._load_panel(frame_number)

        img = np.zeros((self.resolution[1], self.resolution[0], 3), dtype='uint8')

        frame_offset = frame_number % self.frame_offset
        px_offset = round(frame_offset * self.frame_px)

        img[:, :] = self.panel[:, px_offset:self.resolution[0]+px_offset]

        img[:, self.offset-2:self.offset+2, :] = (0, 0, 255)
        return img

    def _load_panel(self, frame_number: int):
        self.panel = np.zeros((self.resolution[1], 3 * self.offset, 3), dtype='uint8')
        self.panel_nb = frame_number // self.frame_offset

        # self.panel[:, :, 0] = random.randint(0, 255)
        # self.panel[:, :, 1] = random.randint(0, 255)
        # self.panel[:, :, 2] = random.randint(0, 255)

        panel_start_ms = self.panel_nb * self.time_range - self.time_range
        panel_end_ms = panel_start_ms + 3 * self.time_range

        panel_start_frame = self.panel_nb * self.frame_offset - self.frame_offset
        panel_end_frame = panel_start_frame + 3 * self.frame_offset

        if self.unit == 'frames':
            annots = get_annots_in_range(self.annots, (panel_start_frame, panel_end_frame))
        else:
            annots = get_annots_in_range(self.annots, (panel_start_ms, panel_end_ms))

        draw_annotations(self.panel, annots, panel_start_frame, panel_end_frame)
