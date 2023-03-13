import abc
import numpy as np


class Displayable(object):

    @abc.abstractmethod
    def get_img(self, frame_number: int) -> np.ndarray:
        pass

    def release(self):
        pass
