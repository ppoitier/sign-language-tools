import numpy as np
import os
import sign_language_tools.pose.mediapipe.utils as utils


def extract_landmarks(path: str, from_dir: bool = False, **kwargs):
    if from_dir:
        results = utils.extract_landmarks_from_dir(path, **kwargs)
    else:
        results = utils.extract_landmarks_from_video_file(path, **kwargs)

    face = np.stack([utils.landmarks_to_np(frame_result.face_landmarks, 468) for frame_result in results])
    pose = np.stack([utils.landmarks_to_np(frame_result.pose_landmarks, 33) for frame_result in results])
    left_hand = np.stack([utils.landmarks_to_np(frame_result.left_hand_landmarks, 21) for frame_result in results])
    right_hand = np.stack([utils.landmarks_to_np(frame_result.right_hand_landmarks, 21) for frame_result in results])

    return face, pose, left_hand, right_hand


def extract_and_save_video_landmarks(video_path: str, video_id: str, dest_path: str, from_dir: bool = False, **kwargs):
    face, pose, left_hand, right_hand = extract_landmarks(video_path, from_dir=from_dir, **kwargs)
    np.save(os.path.join(dest_path, 'face', video_id), face.astype('float16'))
    np.save(os.path.join(dest_path, 'pose', video_id), pose.astype('float16'))
    np.save(os.path.join(dest_path, 'left_hand', video_id), left_hand.astype('float16'))
    np.save(os.path.join(dest_path, 'right_hand', video_id), right_hand.astype('float16'))
