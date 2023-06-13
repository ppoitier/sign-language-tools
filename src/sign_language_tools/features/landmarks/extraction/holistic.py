import numpy as np
import os
import sign_language_tools.utils.mediapipe as mp_utils


def extract_landmarks_from_video(video_path: str, **kwargs):
    results = mp_utils.extract_landmarks_from_video_file(video_path, **kwargs)

    face = np.stack([mp_utils.landmarks_to_np(frame_result.face_landmarks, 478) for frame_result in results])
    pose = np.stack([mp_utils.landmarks_to_np(frame_result.pose_landmarks, 33) for frame_result in results])
    left_hand = np.stack([mp_utils.landmarks_to_np(frame_result.left_hand_landmarks, 21) for frame_result in results])
    right_hand = np.stack([mp_utils.landmarks_to_np(frame_result.right_hand_landmarks, 21) for frame_result in results])

    return face, pose, left_hand, right_hand


def extract_and_save_landmarks_from_video(video_path: str, identifier: str, dest_path: str = './', **kwargs):
    face, pose, left_hand, right_hand = extract_landmarks_from_video(video_path, **kwargs)
    np.save(os.path.join(dest_path, 'face', identifier), face)
    np.save(os.path.join(dest_path, 'pose', identifier), pose)
    np.save(os.path.join(dest_path, 'left_hand', identifier), left_hand)
    np.save(os.path.join(dest_path, 'right_hand', identifier), right_hand)
