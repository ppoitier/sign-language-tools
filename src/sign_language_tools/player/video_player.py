from dataclasses import dataclass
from pathlib import Path
from time import time
from uuid import uuid4

import cv2
import numpy as np
from vidgear.gears import VideoGear

from sign_language_tools.player.drawing_utils.poses import draw_pose
from sign_language_tools.player.drawing_utils.segments import draw_segments


@dataclass()
class Component:
    name: str
    fps: float
    speed: float
    children: list["Component"]


@dataclass()
class VideoComponent(Component):
    filepath: str
    width: int
    height: int


@dataclass()
class EmptyComponent(Component):
    width: int
    height: int
    background_color: tuple[int, int, int]

    def to_frame(self, t: float, parent_frame: np.ndarray | None) -> np.ndarray:
        return np.full(
            (self.height, self.width, 3),
            fill_value=[[self.background_color]],
            dtype="uint8",
        ) if parent_frame is None else parent_frame


@dataclass()
class SkeletonComponent(Component):
    poses: np.ndarray
    frame_lims: np.ndarray
    vertex_lims: np.ndarray
    edges: list[tuple[int, int]] | None
    vertex_color: tuple[int, int, int]
    edge_color: tuple[int, int, int]
    vertex_width: int
    edge_width: int

    def to_frame(self, t: float, parent_frame: np.ndarray | None) -> np.ndarray:
        default_size = self.frame_lims[:, 1] - self.frame_lims[:, 0]
        frame = (
            np.zeros((default_size[1], default_size[0], 3), dtype="uint8")
            if parent_frame is None
            else parent_frame
        )
        n_poses = self.poses.shape[0]
        current_frame = max(0, min(n_poses - 1, round(t * self.fps)))
        current_pose = self.poses[current_frame, :, :2]
        draw_pose(
            frame=frame,
            pose=current_pose,
            edges=self.edges,
            vertex_lims=self.vertex_lims,
            frame_lims=self.frame_lims,
            vertex_color=self.vertex_color,
            edge_color=self.edge_color,
            vertex_width=self.vertex_width,
            edge_width=self.edge_width,
        )
        return frame


@dataclass()
class AnnotationComponent(Component):
    segments: np.ndarray
    labels: list[str] | None
    frame_lims: np.ndarray
    t_lims: np.ndarray
    segment_color: tuple[int, int, int]
    text_color: tuple[int, int, int]
    ticks_color: tuple[int, int, int]
    background_color: tuple[int, int, int] | None
    filled: bool

    def to_frame(self, t: float, parent_frame: np.ndarray | None) -> np.ndarray:
        default_size = self.frame_lims[:, 1] - self.frame_lims[:, 0]
        parent_frame = (
            np.zeros((default_size[1], default_size[0], 3), dtype="uint8")
            if parent_frame is None
            else parent_frame
        )
        draw_segments(
            frame=parent_frame,
            sorted_segments=self.segments,
            t=t,
            t_lims=self.t_lims,
            frame_lims=self.frame_lims,
            labels=self.labels,
            segment_color=self.segment_color,
            text_color=self.text_color,
            background_color=self.background_color,
            filled=self.filled,
            ticks_color=self.ticks_color,
        )
        return parent_frame


class PlaybackInfoComponent(EmptyComponent):
    scale: float = 0.5
    x: int = 10
    y: int = 20

    def to_frame(self, t: float, parent_frame: np.ndarray | None) -> np.ndarray:
        frame = super().to_frame(t, parent_frame)
        current_frame_count = round(t * self.fps)
        cv2.putText(
            frame,
            f"FPS={self.fps} ; Frame={current_frame_count} ; T={t:.2f}s",
            (self.x, self.y),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=self.scale,
            color=(255, 255, 255),
            thickness=1,
            lineType=cv2.LINE_AA,
        )
        return frame


class VideoPlayer:
    def __init__(self):
        self.components: list[Component] = []
        self.default_fps = 24
        self.default_size = (800, 600)

    def attach_video(
        self,
        filepath: str,
        name: str | None = None,
        fps: float | None = None,
        speed: float = 1.0,
    ):
        name = Path(filepath).stem if name is None else name
        cap = cv2.VideoCapture(filepath)
        fps = cap.get(cv2.CAP_PROP_FPS) if fps is None else fps
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.default_fps = fps
        self.default_size = (width, height)
        cap.release()
        self.components.append(
            VideoComponent(
                name=name,
                filepath=filepath,
                fps=fps,
                speed=speed,
                width=width,
                height=height,
                children=[],
            )
        )

    def attach_empty(
        self,
        width: int | None = None,
        height: int | None = None,
        name: str | None = None,
        parent_name: str | None = None,
        fps: float | None = None,
        background_color: tuple[int, int, int] = (0, 0, 0),
    ):
        width = width if width is not None else self.default_size[0]
        height = height if height is not None else self.default_size[1]
        fps = fps if fps is not None else self.default_fps
        name = str(uuid4()) if name is None else name
        component = EmptyComponent(
            name=name,
            fps=fps,
            width=width,
            height=height,
            background_color=background_color,
            speed=1.0,
            children=[],
        )
        if parent_name is None:
            self.components.append(component)
        else:
            parent = self._get_component_by_name(parent_name)
            parent.children.append(component)

    def attach_poses(
        self,
        pose_seq: np.ndarray,
        edges: list[tuple[int, int]] | None = None,
        name: str | None = None,
        parent_name: str | None = None,
        fps: float | None = None,
        speed: float = 1.0,
        x_lim: tuple[int, int] | None = None,
        y_lim: tuple[int, int] | None = None,
        vertex_x_lim: tuple[float, float] = (0.0, 1.0),
        vertex_y_lim: tuple[float, float] = (0.0, 1.0),
        vertex_color: tuple[int, int, int] = (255, 0, 0),
        edge_color: tuple[int, int, int] = (255, 255, 255),
        vertex_width: int = 1,
        edge_width: int = 1,
    ):
        name = str(uuid4()) if name is None else name
        fps = self.default_fps if fps is None else fps
        x_lim = (0, self.default_size[0]) if x_lim is None else x_lim
        y_lim = (0, self.default_size[1]) if y_lim is None else y_lim
        component = SkeletonComponent(
            name=name,
            edges=edges,
            fps=fps,
            speed=speed,
            poses=pose_seq,
            frame_lims=np.array([x_lim, y_lim], dtype="int32"),
            vertex_lims=np.array([vertex_x_lim, vertex_y_lim], dtype="float32"),
            vertex_color=vertex_color,
            edge_color=edge_color,
            vertex_width=vertex_width,
            edge_width=edge_width,
            children=[],
        )
        if parent_name is None:
            self.components.append(component)
        else:
            parent = self._get_component_by_name(parent_name)
            parent.children.append(component)

    def attach_segments(
        self,
        segments: np.ndarray,
        unit: str = "s",
        labels: list[str] | None = None,
        name: str | None = None,
        parent_name: str | None = None,
        fps: float | None = None,
        speed: float = 1.0,
        x_lim: tuple[int, int] = (0, 300),
        y_lim: tuple[int, int] = (0, 200),
        segment_color: tuple[int, int, int] = (0, 255, 0),
        text_color: tuple[int, int, int] = (255, 255, 255),
        background_color: tuple[int, int, int] | None = None,
        ticks_color: tuple[int, int, int] = (255, 255, 255),
        filled: bool = False,
    ):
        name = str(uuid4()) if name is None else name
        fps = self.default_fps if fps is None else fps
        segments = segments[:, :2].astype("float32")
        if unit == "ms":
            segments /= 1000
        elif unit == "frame":
            segments /= fps
        elif unit != "s":
            raise ValueError(f"Unknown unit: '{unit}'.")

        component = AnnotationComponent(
            name=name,
            fps=fps,
            speed=speed,
            segments=segments.astype("float32"),
            labels=labels,
            frame_lims=np.array([x_lim, y_lim], dtype="int32"),
            t_lims=np.array([[-4.0, 4.0], [0.0, 1.0]]),
            segment_color=segment_color,
            text_color=text_color,
            background_color=background_color,
            filled=filled,
            ticks_color=ticks_color,
            children=[],
        )
        if parent_name is None:
            self.components.append(component)
        else:
            parent = self._get_component_by_name(parent_name)
            parent.children.append(component)

    def attach_playback_info(
            self,
            name: str | None = None,
            parent_name: str | None = None,
            fps: float | None = None,
            background_color: tuple[int, int, int] = (0, 0, 0),
            speed: float = 1.0,
            width: int = 300,
            height: int = 100,
    ):
        name = str(uuid4()) if name is None else name
        fps = self.default_fps if fps is None else fps
        component = PlaybackInfoComponent(
            name=name,
            fps=fps,
            speed=speed,
            children=[],
            width=width,
            height=height,
            background_color=background_color,
        )
        if parent_name is None:
            self.components.append(component)
        else:
            parent = self._get_component_by_name(parent_name)
            parent.children.append(component)

    def _get_component_by_name(self, name: str):
        for component in self.components:
            if component.name == name:
                return component
        raise ValueError(f"Component with name '{name}' does not exist.")

    def _sort_components(self):
        # Always display video components before others
        def get_component_order(component):
            if isinstance(component, VideoComponent):
                return 1
            elif isinstance(component, SkeletonComponent):
                return 2
            else:
                return 3

        self.components = list(sorted(self.components, key=get_component_order))

    def _display_component(
        self,
        component: Component,
        stream: VideoGear | None,
        t: float,
        last_timestamp: float,
        current_timestamp: float,
        global_speed: float,
        parent_frame: np.ndarray | None = None,
    ):
        elapsed_ms = current_timestamp - last_timestamp
        min_interval = 1000 / (component.fps * component.speed * global_speed)
        if (parent_frame is None) and (elapsed_ms < min_interval):
            return False, False, parent_frame

        if isinstance(component, VideoComponent):
            assert stream is not None
            frame = stream.read()
        elif (
            isinstance(component, SkeletonComponent)
            or isinstance(component, AnnotationComponent)
            or isinstance(component, EmptyComponent)
        ):
            frame = component.to_frame(t, parent_frame=parent_frame)
        else:
            raise ValueError(f"Unknown component type: {component.name}")
        if frame is None:
            return False, True, parent_frame

        for child_component in component.children:
            _, _, frame = self._display_component(
                component=child_component,
                stream=None,
                t=t,
                last_timestamp=last_timestamp,
                current_timestamp=current_timestamp,
                global_speed=global_speed,
                parent_frame=frame,
            )

        if parent_frame is None:
            cv2.imshow(component.name, frame)

        return True, False, frame

    def play(self, speed: float = 1.0):
        self._sort_components()

        video_streams = {}
        last_timestamps = {}
        current_frame_nbs = {}

        for component in self.components:
            if isinstance(component, VideoComponent):
                video_streams[component.name] = VideoGear(
                    source=component.filepath
                ).start()
            last_timestamps[component.name] = time() * 1000
            current_frame_nbs[component.name] = 0

        paused = False
        should_stop = False
        global_speed = speed

        while not should_stop:
            if not paused:
                should_stop = True
                for component in self.components:
                    current_timestamp = time() * 1000
                    current_t = current_frame_nbs[component.name] / component.fps
                    is_drawn, should_stop, _ = self._display_component(
                        component=component,
                        stream=video_streams.get(component.name),
                        t=current_t,
                        last_timestamp=last_timestamps[component.name],
                        current_timestamp=current_timestamp,
                        global_speed=global_speed,
                    )
                    if is_drawn:
                        last_timestamps[component.name] = current_timestamp
                        current_frame_nbs[component.name] += 1
            key = cv2.waitKeyEx(1)
            # stop if 'q' key is pressed
            if key == ord("q"):
                should_stop = True
            # pause (or unpause) if 'space' key is pressed
            elif key == ord(" "):
                paused = not paused
            # right arrow
            elif key == 65363:
                if global_speed <= 16:
                    global_speed *= 2
            # left arrow
            elif key == 65361:
                if global_speed > speed:
                    global_speed //= 2

        # Clear the memory, close the windows and close the video streams
        cv2.destroyAllWindows()
        for stream in video_streams.values():
            stream.stop()
