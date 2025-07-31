from vidgear.gears import CamGear
import cv2
import numpy as np
import os
import mediapipe as mp
from tqdm import tqdm


mp_holistic = mp.solutions.holistic


def _mediapipe_output_to_numpy_arrays(output, expected_nb_of_landmarks: int):
    if output is None:
        return np.full((expected_nb_of_landmarks, 3), fill_value=np.nan, dtype='float16')
    return np.array([(lm.x, lm.y, lm.z) for lm in output.landmark], dtype='float16')


def extract_poses_from_video(
    video_path: str,
    region_of_interest: tuple[int, int, int, int] = None,
    show_progress: bool = False,
    options=None,
):
    if not os.path.isfile(video_path):
        raise FileNotFoundError("Video file not found.")

    landmarks: dict[str, list[np.ndarray]] = {
        "face": [],
        "pose": [],
        "left_hand": [],
        "right_hand": [],
    }
    capture = CamGear(source=video_path).start()
    n_frames = int(capture.stream.get(cv2.CAP_PROP_FRAME_COUNT))
    progress_bar = tqdm(range(n_frames), unit="frames", disable=not show_progress)

    if options is None:
        options = {
            'static_image_mode': False,
            'model_complexity': 1,
            'refine_face_landmarks': False,
            'smooth_landmarks': True,
            'min_detection_confidence': 0.2,
            'min_tracking_confidence': 0.2,
            'enable_segmentation': False,
            'smooth_segmentation': False,
        }

    with mp_holistic.Holistic(**options) as holistic:
        for frame_nb in progress_bar:
            # time_stamp_ms = int((frame_nb / frame_rate) * 1000)
            frame = capture.read()
            if frame is None:
                progress_bar.write(f"Cannot read frame {frame_nb}.")
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if region_of_interest is not None:
                frame = frame[
                    region_of_interest[2] : region_of_interest[3],
                    region_of_interest[0] : region_of_interest[1],
                ].copy()
            mp_results = holistic.process(frame)
            landmarks["face"].append(_mediapipe_output_to_numpy_arrays(mp_results.face_landmarks, 468))
            landmarks["pose"].append(_mediapipe_output_to_numpy_arrays(mp_results.pose_landmarks, 33))
            landmarks["left_hand"].append(_mediapipe_output_to_numpy_arrays(mp_results.left_hand_landmarks, 21))
            landmarks["right_hand"].append(_mediapipe_output_to_numpy_arrays(mp_results.right_hand_landmarks, 21))
            cv2.waitKey(1)
    cv2.destroyAllWindows()
    capture.stop()
    landmarks["face"] = np.stack(landmarks["face"], axis=0)
    landmarks["pose"] = np.stack(landmarks["pose"], axis=0)
    landmarks["left_hand"] = np.stack(landmarks["left_hand"], axis=0)
    landmarks["right_hand"] = np.stack(landmarks["right_hand"], axis=0)
    return landmarks


if __name__ == "__main__":
    extract_poses_from_video(
        video_path="D:/data/sign-languages/nii_jsl/sample.mp4",
        region_of_interest=(0, 397, 0, 305),
        show_progress=True,
    )
