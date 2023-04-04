import cv2
from .video import Video
from .images import Images
from .skeleton import Skeleton
from .annotations import Annotations
from sign_language_tools.features.landmarks.edges.base import HAND_EDGES, POSE_EDGES
from sign_language_tools.features.landmarks.edges.facemesh import FACEMESH_CONTOURS
from typing import Optional, Literal
import numpy as np
import pandas as pd
import os


class VideoPlayer:
    """Video player able to display video with annotations or landmarks information.

    The video player is a versatile video player that allows to visualise sign language landmarks in mediapipe format
    along with a video stream and annotations.
    """
    def __init__(
            self,
            root: Optional[str] = None,
            screenshot_dir: Optional[str] = None,
            fps: int = 24,
    ):
        """Creation of a new video player

        Args:
            root: An optional root path prepended to all.
            screenshot_dir: An optional folder path in which screenshot should be saved.
            fps: The framerate of the information to display (default = 24).
        """
        self.resolution = (756, 512)
        self.current_frame = 0
        self.frame_count = 0
        self.fps = fps
        self.speed = 1.0

        self.root = root
        self.screenshot_dir = screenshot_dir
        self.video: Optional[Video | Images] = None
        self.skeleton: Optional[Skeleton] = None
        self.annotations: list[Annotations] = []

        self.stop = False
        self.pause = False
        self.last_images = {}

        self.isolate_video = False
        self.isolate_skeleton = False
        self.crop = (0.0, 1.0, 0.0, 1.0)

    def attach_video(self, video_path: str):
        """Attach a video file to the video player

        Only one video could be attached to a video player. The player use
        `opencv-python` to render the video.

        Args:
            video_path: The path of the video file to attach.
        """

        self.video = Video(self._get_path(video_path))
        self.resolution = (self.video.width, self.video.height)
        self.frame_count = self.video.frame_count
        self.fps = self.video.fps

    def attach_image_dir(self, dir_path: str, extension: str = 'png', fps: int = 25):
        """Attach an image folder to the video player.

        The images in the folder are used as video frame in the video player. The images are
        displayed in alphabetical order.

        Only one video could be attached to a video player. The player use
        `opencv-python` to render the video.

        Args:
            dir_path: The path of the folder containing the images.
            extension: The file extension of the images (default = 'png').
            fps: The framerate of the video to display.
        """
        self.video = Images(self._get_path(dir_path), extension=extension)
        self.frame_count = self.video.frame_count
        self.fps = fps

    def attach_landmarks(self, name: str, landmarks: np.ndarray, connections=None, show_vertices=True):
        """Attach a set of landmarks to the video player.

        Each set of landmark is identified by its name.
        The video player is able to display multiple sets of landmarks in the same video.

        Args:
            name: The name of the set of landmarks.
            landmarks: The tensor containing the signal values of each landmark.
            connections: An optional set of connections between the landmarks (default = None).
            show_vertices: If `True`, the vertices of the landmarks are displayed.
                Otherwise, they are hidden (default = `True`).

        Shape:
            landmarks: of the shape (T, N, D) where T is the number of frames, N the number of landmarks (vertices)
            and D the dimension of each coordinate (only the two first coordinates are shown).

        Author:
            ppoitier (v1 01.04.2023)
        """
        self._add_landmarks(name, landmarks, connections, show_vertices)

    def attach_pose(self, pose: np.ndarray):
        """Attach pose landmarks of MediaPipe to the video player along with the corresponding edges.

        Be careful that this method overrides the set of landmarks called `pose`.

        Args:
            pose: The tensor containing the signal values of the pose landmarks.

        Shape:
            landmarks: of the shape (T, N, D) where T is the number of frames, N the number of landmarks (vertices)
            and D the dimension of each coordinate (only the two first coordinates are shown).

        Author:
            ppoitier (v1 01.04.2023)
        """
        self.attach_landmarks('pose', pose, POSE_EDGES)

    def attach_pose_from_csv(self, csv_path: str):
        """Attach pose landmarks of MediaPipe to the video player along with the corresponding edges.

        Be careful that this method overrides the set of landmarks called `pose`.

        Args:
            csv_path: Path of the CSV file containing the coordinates of the signal values of the pose landmarks.

        Author:
            ppoitier (v1 01.04.2023)
        """
        pose = pd.read_csv(self._get_path(csv_path)).values
        self.attach_pose(pose)

    def attach_hand(self, landmarks: np.ndarray, hand: str):
        self.attach_landmarks(f'{hand}_hand', landmarks, HAND_EDGES)

    def attach_hand_from_csv(self, csv_path: str, hand: str):
        landmarks = pd.read_csv(self._get_path(csv_path)).values
        self.attach_hand(landmarks, hand)

    def attach_hands(self, hands: np.ndarray):
        if len(hands.shape) == 2:
            left = hands[:, :42]
            right = hands[:, 42:]
        else:
            left = hands[:, :21]
            right = hands[:, 21:]
        self.attach_hand(left, 'left')
        self.attach_hand(right, 'right')

    def attach_hands_from_csv(self, csv_path: str):
        hands = pd.read_csv(self._get_path(csv_path)).values
        self.attach_hands(hands)

    def attach_face(self, face: np.ndarray, mesh: bool = False):
        # TODO : add mesh
        self.attach_landmarks('face', face, FACEMESH_CONTOURS, show_vertices=False)

    def attach_face_from_csv(self, csv_path: str, mesh: bool = False):
        face = pd.read_csv(self._get_path(csv_path)).values
        self.attach_face(face, mesh)

    def attach_annotations_from_csv(self, csv_path: str, *args, **kwargs):
        annots = pd.read_csv(self._get_path(csv_path)).iloc[:, [0, 1, 2]]
        annots.columns = ['start', 'end', 'label']
        self.attach_annotations(annots, *args, **kwargs)

    def attach_annotations(self, annots: pd.DataFrame, name: str, unit: str = 'ms'):
        self.annotations.append(Annotations(annots, name, fps=self.fps, unit=unit))

    def set_crop(self, x: tuple[float, float] = (0, 1), y: tuple[float, float] = (0, 1)):
        """Crop the viewport of the video player.

        Args:
            x: The relative x-axis range (start, end) to use. Example: `x=(0.4, 0.7)`.
            y: The relative y-axis range (start, end) to use. Example: `y=(0.2, 0.5)`.

        Author:
            ppoitier (v1 01.04.2023)
        """
        assert 0 <= x[0] <= 1
        assert 0 <= x[1] <= 1
        assert 0 <= y[0] <= 1
        assert 0 <= y[1] <= 1
        self.crop = (x[0], x[1], y[0], y[1])

    def isolate(self, element: Literal['video', 'skeleton']):
        """Isolate an element out of the main window, into a new window.

        Args:
            element: The element that is isolated:
                - `video` to isolate the original video.
                - `skeleton` to isolate the landmarks.

        Author:
            ppoitier (v1 01.04.2023)
        """
        if element == 'video':
            self.isolate_video = True
        elif element == 'skeleton':
            self.isolate_skeleton = True

    def set_speed(self, new_speed: float):
        """Change the playback speed of the video player.

        Example:
            ```
            set_speed(2.0)  # Double the original playback speed of the video player
            ```

        Args:
            new_speed: New relative playback speed (must be positive)

        Author:
            ppoitier (v1 01.04.2023)
        """
        assert 0 < new_speed, 'Speed must be positive.'
        self.speed = new_speed

    def play(self):
        """Start the video player and display all the attached elements.

        The player use `opencv-python` to render the video.

        Author:
            ppoitier (v1 01.04.2023)
        """
        self.current_frame = 0
        frame_duration = round(1000/self.fps)
        delay = round(frame_duration / self.speed)

        while (self.current_frame < self.frame_count) and not self.stop:
            if not self.pause:
                self._next_frame(self.current_frame)
                self.current_frame = self.current_frame + 1

            key_code = cv2.waitKeyEx(delay)
            self._user_action(key_code)

        self._stop()

    # ---- Private methods

    def _next_frame(self, frame_nb: int):
        self.last_images = {}
        if self.video is not None:
            img = self.video.get_img(frame_nb)
            if self.isolate_video:
                self._show_frame('Video (isolated)', img.copy(), frame_nb)

            if self.skeleton is not None:
                self.skeleton.draw(img, frame_nb)
                if self.isolate_skeleton:
                    self._show_frame('Skeleton (isolated)', self.skeleton.get_img(frame_nb), frame_nb)
        elif self.skeleton is not None:
            img = self.skeleton.get_img(frame_nb)
        else:
            img = np.zeros((512, 756, 3), dtype='uint8')

        self._show_frame('Video Player', img, frame_nb)

        for annot in self.annotations:
            cv2.imshow(annot.name, annot.get_img(frame_nb))

    def _show_frame(self, window_title: str, frame: np.ndarray, frame_nb: int):
        if self.crop is not None:
            x_start, x_end, y_start, y_end = self.crop
            x_start = int(x_start * frame.shape[1])
            x_end = int(x_end * frame.shape[1])
            y_start = int(y_start * frame.shape[0])
            y_end = int(y_end * frame.shape[0])
            frame = frame[y_start:y_end, x_start:x_end]
        cv2.imshow(window_title, frame)
        self.last_images[window_title] = (frame_nb, frame)

    def _user_action(self, key_code):
        if key_code == ord('q'):
            self._stop()
        if key_code == ord('s'):
            self._screenshot()
        elif key_code == 27:        # Escape
            self._stop()
        elif key_code == 2424832:   # Left arrow
            self._goto(self.current_frame - self.fps * 10)
        elif key_code == 2555904:   # Right arrow
            self._goto(self.current_frame + self.fps * 10)
        elif key_code == 32:        # Space bar
            self.pause = not self.pause

    def _goto(self, frame_nb: int):
        frame_nb = max(0, min(frame_nb, self.frame_count))
        self.current_frame = frame_nb

    def _stop(self):
        self.stop = True

        if self.video is not None:
            self.video.release()

        cv2.destroyAllWindows()

    def _screenshot(self):
        if self.screenshot_dir is None:
            return

        _dir = self.screenshot_dir
        if os.path.isdir(os.path.join(self.root, _dir)):
            _dir = os.path.join(self.root, _dir)

        if not os.path.isdir(_dir):
            print('Could not save frames. Directory not found:', _dir)
            return

        for window_title in self.last_images:
            frame_nb, frame = self.last_images[window_title]
            filepath = os.path.normpath(os.path.join(_dir, f'{frame_nb}_{window_title}.jpg'))
            cv2.imwrite(filepath, frame)
            print('Saved:', filepath)

    def _add_landmarks(self, name: str, landmarks: np.ndarray, connections, show_vertices: bool):
        if self.skeleton is None:
            self.skeleton = Skeleton(mesh=False, resolution=self.resolution)
        self.skeleton.add_landmarks(name, landmarks, connections, show_vertices)

        landmarks_count = self.skeleton.n_poses
        if self.video is not None:
            assert self.frame_count == landmarks_count, \
                f'Different number of frames ({self.frame_count}) and landmarks ({landmarks_count}).'
        else:
            self.frame_count = landmarks_count

    def _get_path(self, path: str):
        if self.root is not None:
            return os.path.join(self.root, path)
        return path
