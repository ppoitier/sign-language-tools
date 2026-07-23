import numpy as np


def from_json_to_np(pose_data: dict, dtype='float16') -> dict[str, np.ndarray]:
    """ Convert OpenPose JSON data to a Numpy array.
    Caution! This function does not handle multiple signers nor 3D landmarks.
    Tested on OpenPose 1.3 2D features.

    Args:
        pose_data: dictionary containing OpenPose JSON data.
        dtype: type used for each coordinate value (default=float16).

    Returns:
        pose_array: dictionary of numpy arrays of shape (N, D),
            where N is the number of landmarks and D=2 the coordinates dimension.
            Keys of the dictionary correspond to:
                - pose -> 25 landmarks
                - face -> 70 landmarks
                - left_hand -> 21 landmarks
                - right_hand -> 21 landmarks
    """
    pose_arrays = {
        'pose': np.full((25, 2), fill_value=np.nan, dtype=dtype),
        'face': np.full((70, 2), fill_value=np.nan, dtype=dtype),
        'left_hand': np.full((21, 2), fill_value=np.nan, dtype=dtype),
        'right_hand': np.full((21, 2), fill_value=np.nan, dtype=dtype),
    }

    # Check if there is not any signer
    if len(pose_data['people']) < 1:
        return pose_arrays
    pose_data = pose_data['people'][0]

    # our key -> open pose key
    key_matching = {
        'pose': 'pose_keypoints_2d',
        'face': 'face_keypoints_2d',
        'left_hand': 'hand_left_keypoints_2d',
        'right_hand': 'hand_right_keypoints_2d',
    }
    for key in key_matching:
        data = pose_data[key_matching[key]]
        if len(data) > 0:
            pose_arrays[key] = np.array(data).reshape(-1, 3)[:, :2]

    return pose_arrays
