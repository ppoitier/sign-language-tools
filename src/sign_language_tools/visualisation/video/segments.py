import numpy as np
import pandas as pd
from sign_language_tools.visualisation.video.displayable import Displayable
from sign_language_tools.visualisation.video.cv.annotations import draw_segments


def get_segments_in_range(segments: pd.DataFrame, full_range: tuple[int, int]) -> pd.DataFrame:
    start, end = full_range
    return segments.loc[(segments['start'] <= end) & (segments['end'] >= start)]


class Segments(Displayable):

    def __init__(
            self,
            segments: pd.DataFrame,
            name: str,
            unit: str,
            resolution: tuple[int, int] = (512, 256),
            time_range: int = 3000,
            fps: int = 50,
    ):
        self.name = name

        self.segments = segments.copy()
        self.resolution = resolution
        self.offset = resolution[0]//2
        self.time_range = time_range

        self.frame_duration = round(1000/fps)
        self.frame_offset = round(time_range/self.frame_duration)
        self.frame_px = self.offset/self.frame_offset

        self.panel = None
        self.panel_nb = -1
        self.unit = unit

        if self.unit == 'ms':
            self.segments.loc[:, ['start', 'end']] //= self.frame_duration
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

        panel_start_frame = self.panel_nb * self.frame_offset - self.frame_offset
        panel_end_frame = panel_start_frame + 3 * self.frame_offset
        panel_segments = get_segments_in_range(self.segments, (panel_start_frame, panel_end_frame))

        draw_segments(self.panel, panel_segments, panel_start_frame, panel_end_frame)
