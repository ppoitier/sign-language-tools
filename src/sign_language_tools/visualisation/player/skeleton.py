import numpy as np
from .displayable import Displayable
from .cv.landmarks import draw_landmarks


class Skeleton(Displayable):

    def __init__(
            self,
            resolution: tuple[int, int],
            mesh: bool = False,
    ):
        self.resolution = resolution

        self.n_poses = 0
        self.landmarks = {}

        self.mesh = mesh

    def add_landmarks(self, name: str, landmarks: np.ndarray, connections, show_vertices: bool):
        # landmarks: (T, V, D) or (T, VxD)
        if len(landmarks.shape) == 2:
            landmarks = landmarks.reshape((landmarks.shape[0], -1, 2))
        self.n_poses = landmarks.shape[0]
        self.landmarks[name] = landmarks, connections, show_vertices

    def get_img(self, frame_number: int) -> np.ndarray:
        width, height = self.resolution
        img = np.zeros((height, width, 3), dtype='uint8')
        self.draw(img, frame_number)
        return img

    def draw(self, img: np.ndarray, frame_number: int):

        for name in self.landmarks:
            landmarks, connections, show_vertices = self.landmarks[name]
            draw_landmarks(img, landmarks[frame_number], connections, show_vertices=show_vertices)

        # if self.pose is not None:
        #     draw_pose_landmarks(img, self.pose[frame_number])
        #
        # if self.hands is not None:
        #     draw_hands_landmarks(img, self.hands[frame_number])
        #
        # if self.face is not None:
        #     if self.mesh:
        #         draw_face_mesh(img, self.face[frame_number])
        #     else:
        #         draw_face_landmarks(img, self.face[frame_number])
