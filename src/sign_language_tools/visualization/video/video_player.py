import os
from typing import Optional, Literal, Union

import cv2
import numpy as np
import pandas as pd

from sign_language_tools.visualization.video.video import Video
from sign_language_tools.visualization.video.images import Images
from sign_language_tools.visualization.video.poses import Poses
from sign_language_tools.visualization.video.segments import Segments


class VideoPlayer:
    """
    The video player is a versatile video player that allows to visualise sign language landmarks
    along with a video stream and annotations.
    Landmarks must be in the Mediapipe or OpenPose format.

    Controls:
    - Q or Escape: quit the player
    - Space: pause
    - S: Screenshot
    - Left-arrow: 10 seconds forward
    - Right-arrow: 10 seconds backward

    Author:
        v0.0.1 - ppoitier
    """

    def __init__(
        self,
        root: Optional[str] = None,
        screenshot_dir: Optional[str] = None,
        fps: int = 24,
    ):
        """
        Creation of a new video player.

        Args:
            root: An optional root path prepended to all paths.
            screenshot_dir: An optional folder path in which screenshot should be saved.
            fps: The framerate of the video player (default = 24).
                This is automatically changed if a video is attached to the video player.
                Therefore, we recommend you to only manually specify it when no video stream is used.
        """
        self.resolution = (756, 512)
        self.current_frame = 0
        self.frame_count = 0
        self.fps = fps
        self.speed = 1.0

        self.root = root
        self.screenshot_dir = screenshot_dir
        self.video: Optional[Union[Video, Images]] = None
        self.poses: Optional[Poses] = None
        self.segmentations: list[Segments] = []

        self.stop = False
        self.pause = False
        self.last_images = {}

        self.isolate_video = False
        self.isolate_pose = False
        self.crop = (0.0, 1.0, 0.0, 1.0)

    def attach_video(self, video_path: str):
        """
        Attach a video file to the video player.

        Only one video could be attached to a video player. The player use
        `opencv-python` to render the video.

        Args:
            video_path: The path of the video file to attach.
        """

        self.video = Video(self._get_path(video_path))
        self.resolution = (self.video.width, self.video.height)
        self.frame_count = self.video.frame_count
        self.fps = self.video.fps

    def attach_image_dir(
        self, dir_path: str, extension: str = "png", fps: Optional[int] = None
    ):
        """
        Attach an image folder to the video player.

        The images in the folder are used as video frames in the video player. They are
        displayed in alphabetical order.

        Only one video could be attached to a video player. The player use
        `opencv-python` to render the video.

        Args:
            dir_path: The path of the folder containing the images.
            extension: The file extension of the images (default = 'png').
            fps: If specified, change the framerate of the video player. Default=None
        """
        self.video = Images(self._get_path(dir_path), extension=extension)
        self.frame_count = self.video.frame_count
        if fps is not None:
            self.fps = fps

    def attach_pose(
        self,
        name: str,
        pose_sequence: np.ndarray,
        connections=None,
        show_vertices: bool = True,
        vertex_color=(0, 0, 255),
        edge_color=(0, 255, 0),
    ):
        """
        Attach a set of landmarks to the video player.

        Each set of landmark is identified by its name.
        The video player is able to display multiple sets of landmarks in the same video.
        However, they must have different names. Otherwise, the previous landmarks are replaced.

        Args:
            name: The name of the set of landmarks.
            pose_sequence: The tensor containing the signal values of each landmark.
                Must be of shape (T, L, 2) or (T, L, 3) where T is the number of frames and L the number of landmarks.
            connections: An optional set of connections between the landmarks. Default=None
            show_vertices: If true, the vertices of the landmarks are displayed.
                Otherwise, they are hidden. Default=True
            vertex_color: The color of each vertex in the BGR format (0 <= color <= 255) of OpenCV.
            edge_color: The color of each edge in the BGR format (0 <= color <= 255) of OpenCV.

        Shape:
            landmarks: of the shape (T, L, D) where T is the number of frames, L the number of landmarks (vertices)
            and D the dimension of each coordinate (only the two first coordinates are shown). Typically, D=2 or D=3.
        """
        self._add_pose(
            name, pose_sequence, connections, show_vertices, vertex_color, edge_color
        )

    def attach_segments(
        self, name: str, segments: Union[pd.DataFrame, np.ndarray], unit: str = "ms"
    ):
        if isinstance(segments, np.ndarray):
            segments = pd.DataFrame(
                {
                    "start": pd.Series(segments[:, 0], dtype="int32"),
                    "end": pd.Series(segments[:, 1], dtype="int32"),
                    "label": pd.Series(segments[:, 2]),
                }
            )
        self.segmentations.append(Segments(segments, name, fps=self.fps, unit=unit))

    def set_crop(
        self, x: tuple[float, float] = (0, 1), y: tuple[float, float] = (0, 1)
    ):
        """
        Crop the viewport of the video player.

        Default viewport is x=(0, 1) and y=(0, 1).
        x: left -> right
        y: top -> bottom

        Args:
            x: The relative x-axis range (start, end) to use. Example: `x=(0.4, 0.7)`.
            y: The relative y-axis range (start, end) to use. Example: `y=(0.2, 0.5)`.
        """
        assert 0 <= x[0] <= 1
        assert 0 <= x[1] <= 1
        assert 0 <= y[0] <= 1
        assert 0 <= y[1] <= 1
        self.crop = (x[0], x[1], y[0], y[1])

    def isolate(self, element: Literal["video", "pose"]):
        """
        Isolate an element out of the main window, into a new window.

        Args:
            element: The element that is isolated:
                - `video` to isolate the original video.
                - `pose` to isolate the landmarks.
        """
        if element == "video":
            self.isolate_video = True
        elif element == "pose":
            self.isolate_pose = True

    def set_speed(self, new_speed: float):
        """
        Change the playback speed of the video player.

        Example:
            ```
            set_speed(2.0)  # Double the original playback speed of the video player
            ```

        Args:
            new_speed: New relative playback speed (must be positive).
        """
        assert 0 < new_speed, "Speed must be positive."
        self.speed = new_speed

    def play(self):
        """
        Start the video player and display all the attached elements.
        The player use `opencv-python` to render the video.
        """
        self.current_frame = 0
        frame_duration = round(1000 / self.fps)
        delay = round(frame_duration / self.speed)

        while (self.current_frame < self.frame_count) and not self.stop:
            if not self.pause:
                self._next_frame(self.current_frame)
                self.current_frame = self.current_frame + 1

            key_code = cv2.waitKeyEx(delay)
            self._user_action(key_code)

        self._stop()

    def save(self, path: str):
        """
        Save the main player window as a mp4 video file.
        Example:
            ```
            save("./results.mp4")  # Save the video in results.mp4
            ```

        Args:
            path: The relative path to the video file.
        """
        self.current_frame = 0

        frame_duration = round(1000 / self.fps)
        delay = round(frame_duration / self.speed)

        fourcc = cv2.VideoWriter_fourcc("m", "p", "4", "v")
        out = cv2.VideoWriter(path, fourcc, self.fps, self.resolution)

        while self.current_frame < self.frame_count:
            frame = self._draw_frame(self.current_frame)
            out.write(frame)
            self.current_frame = self.current_frame + 1

        out.release()

    # ---- Private methods

    def _draw_frame(self, frame_nb: int):
        self.last_images = {}

        if self.video is not None:
            img = self.video.get_img(frame_nb)

            if self.poses is not None:
                self.poses.draw(img, frame_nb)

        elif self.poses is not None:
            img = self.poses.get_img(frame_nb)

        else:
            img = np.zeros((512, 756, 3), dtype="uint8")

        return self._get_frame(img)

    def _next_frame(self, frame_nb: int):
        self.last_images = {}
        if self.video is not None:
            img = self.video.get_img(frame_nb)
            if self.isolate_video:
                self._show_frame("Video (isolated)", img.copy(), frame_nb)

            if self.poses is not None:
                self.poses.draw(img, frame_nb)
                if self.isolate_pose:
                    self._show_frame(
                        "Pose (isolated)", self.poses.get_img(frame_nb), frame_nb
                    )
        elif self.poses is not None:
            img = self.poses.get_img(frame_nb)
        else:
            img = np.zeros((512, 756, 3), dtype="uint8")

        self._show_frame("Video Player", img, frame_nb)

        for segments in self.segmentations:
            cv2.imshow(segments.name, segments.get_img(frame_nb))

    def _show_frame(self, window_title: str, frame: np.ndarray, frame_nb: int):
        if self.crop is not None:
            self._crop_frame(frame)

        cv2.imshow(window_title, frame)
        self.last_images[window_title] = (frame_nb, frame)

    def _get_frame(self, frame: np.ndarray):
        if self.crop is not None:
            self._crop_frame(frame)

        return frame

    def _crop_frame(self, frame: np.ndarray):
        x_start, x_end, y_start, y_end = self.crop
        x_start = int(x_start * frame.shape[1])
        x_end = int(x_end * frame.shape[1])
        y_start = int(y_start * frame.shape[0])
        y_end = int(y_end * frame.shape[0])
        return frame[y_start:y_end, x_start:x_end]

    def _user_action(self, key_code):
        if key_code == ord("q"):
            self._stop()
        elif key_code == ord("s"):
            self._screenshot()
        # Escape
        elif key_code == 27:
            self._stop()
        # Left arrow
        elif key_code == 65361 or key_code == 2424832:
            self._goto(self.current_frame - self.fps * 10)
        # Right arrow
        elif key_code == 65363 or key_code == 2555904:
            self._goto(self.current_frame + self.fps * 10)
        # Space bar
        elif key_code == 32:
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
            print("Could not save frames. Directory not found:", _dir)
            return

        for window_title in self.last_images:
            frame_nb, frame = self.last_images[window_title]
            filepath = os.path.normpath(
                os.path.join(_dir, f"{frame_nb}_{window_title}.jpg")
            )
            cv2.imwrite(filepath, frame)
            print("Saved:", filepath)

    def _add_pose(
        self,
        name: str,
        landmarks: np.ndarray,
        connections,
        show_vertices: bool,
        vertex_color: tuple[int, int, int],
        edge_color: tuple[int, int, int],
    ):
        if self.poses is None:
            self.poses = Poses(resolution=self.resolution)
        self.poses.add_pose(
            name, landmarks, connections, show_vertices, vertex_color, edge_color
        )

        n_poses = self.poses.n_poses
        if self.video is not None:
            assert (
                self.frame_count == n_poses
            ), f"Different number of frames ({self.frame_count}) and poses ({n_poses})."
        else:
            self.frame_count = n_poses

    def _get_path(self, path: str):
        if self.root is not None:
            return os.path.join(self.root, path)
        return path
