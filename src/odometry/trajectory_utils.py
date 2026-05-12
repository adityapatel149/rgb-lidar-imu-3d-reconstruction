import numpy as np
from scipy.spatial.transform import Rotation as R



def make_transform(rotation_matrix=None, translation=None):
    T = np.eye(4, dtype=np.float64)

    if rotation_matrix is not None:
        T[:3,:3] = rotation_matrix
    
    if translation is not None:
        T[:3,3] = translation

    return T



def pose_row_to_matrix(row):
    x, y, z = row["x"], row["y"], row["z"]
    roll, pitch, yaw = row["roll"], row["pitch"], row["yaw"]

    rot = R.from_euler(
        "xyz",
        [roll, pitch, yaw],
        degrees = True,
    ).as_matrix()

    return make_transform(rot, np.array([x,y,z], dtype=np.float64))



def relative_to_first(poses):
    T0_inv = np.linalg.inv(poses[0])
    return [T0_inv @ T for T in poses]



def poses_to_xyz(poses):
    return np.array([T[:3, 3] for T in poses], dtype=np.float64)



def normalize_quaternion(q):
    q = np.asarray(q, dtype=np.float64)
    norm = np.linalg.norm(q)
    
    if norm < 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)

    return q / norm