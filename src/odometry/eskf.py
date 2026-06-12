import numpy as np
from scipy.spatial.transform import Rotation as R



def skew(v):
    x, y, z = v
    return np.array(
        [
            [0, -z, y],
            [z, 0, -x],
            [-y, x, 0],
        ], dtype=np.float64    
    )



def exp_so3(w):
    return R.from_rotvec(w).as_matrix()



def log_so3(R_mat):
    return R.from_matrix(R_mat).as_rotvec()



def make_transform(R_mat, p):
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R_mat
    T[:3, 3] = p
    return T



class ErrorStateKalmanFilter:
    """
    Nominal state:
        p   : position in world
        v   : velocity in world
        R   : body-to-world rotation
        b_a : accel bias
        b_g : gyro bias

    Error state:
        dx = [dp, dv, dtheta, db_a, db_g] 15x1
    """

    def __init__(self, initial_pose=None, initial_velocity=None, gravity=np.array([0.0, 0.0, -9.81])):
        if initial_pose is None:
            initial_pose = np.eye(4, dtype=np.float64)

        self.p = initial_pose[:3, 3].astype(np.float64)
        self.v = np.asarray(initial_velocity, dtype=np.float64) if initial_velocity is not None else np.zeros(3, dtype=np.float64)
        self.R = initial_pose[:3, :3].astype(np.float64)
        self.b_a = np.zeros(3, dtype=np.float64)
        self.b_g = np.zeros(3, dtype=np.float64)

        self.g = gravity.astype(np.float64)

        self.P = np.eye(15, dtype=np.float64) * 1e-2
        
        self.P[0:3, 0:3] *= 1e-4      # position known relative to start
        self.P[3:6, 3:6] *= 10.0      # velocity uncertain
        self.P[6:9, 6:9] *= 1e-3      # orientation moderately known
        self.P[9:12, 9:12] *= 0.1     # accel bias uncertain
        self.P[12:15, 12:15] *= 0.01  # gyro bias uncertain

        self.sigma_acc = 0.15
        self.sigma_gyro = 0.02
        self.sigma_acc_bias = 0.001
        self.sigma_gyro_bias = 0.0005



    def pose_matrix(self):
        return make_transform(self.R, self.p)



    def propagate(self, acc_meas, gyro_meas, dt):
        dt = float(np.clip(dt, 1e-4, 0.1))
        acc = acc_meas - self.b_a
        gyro = gyro_meas - self.b_g

        R_prev = self.R.copy()
        v_prev = self.v.copy()

        self.R = self.R @ exp_so3(gyro*dt)

        acc_world = R_prev @ acc + self.g

        self.p = self.p + v_prev * dt + 0.5 * acc_world * dt * dt
        self.v = self.v + acc_world * dt

        F = np.eye(15, dtype=np.float64)
        F[0:3, 3:6] = np.eye(3) * dt
        F[3:6, 6:9] = -R_prev @ skew(acc) * dt
        F[3:6, 9:12] = -R_prev * dt
        F[6:9, 6:9] = exp_so3(-gyro * dt)
        F[6:9, 12:15] = -np.eye(3) * dt

        Q = np.zeros((15,15), dtype=np.float64)
        Q[3:6, 3:6] = np.eye(3) * self.sigma_acc**2 * dt
        Q[6:9, 6:9] = np.eye(3) * self.sigma_gyro**2 * dt
        Q[9:12, 9:12] = np.eye(3) * self.sigma_acc_bias**2 * dt
        Q[12:15, 12:15] = np.eye(3) * self.sigma_gyro_bias**2 * dt

        self.P = F @ self.P @ F.T + Q




    # # Smaller values for posnoise and rotnoise mean trust ICP more, larger values mean trust IMU more.
    # # Here I am trusting icp more for position, imu more for rotational
    def update_pose(
        self,
        T_world_body_meas,
        pos_noise=0.35,
        rot_noise=0.65,
        update_velocity=True,
        update_bias=False,
       # gate_threshold=16.8,  # chi-square 6 dof, ~99%
    ):
        p_meas = T_world_body_meas[:3, 3]
        R_meas = T_world_body_meas[:3, :3]

        r_p = p_meas - self.p
        r_R = log_so3(self.R.T @ R_meas)
        residual = np.hstack([r_p, r_R])

        H = np.zeros((6, 15), dtype=np.float64)
        H[0:3, 0:3] = np.eye(3)
        H[3:6, 6:9] = np.eye(3)

        R_cov = np.zeros((6, 6), dtype=np.float64)
        R_cov[0:3, 0:3] = np.eye(3) * pos_noise**2
        R_cov[3:6, 3:6] = np.eye(3) * rot_noise**2

        S = H @ self.P @ H.T + R_cov

        K = self.P @ H.T @ np.linalg.inv(S)
        dx = K @ residual

        # Debug/safety options
        if not update_velocity:
            dx[3:6] = 0.0

        if not update_bias:
            dx[9:12] = 0.0
            dx[12:15] = 0.0

        self.inject_error(dx)

        I = np.eye(15)
        self.P = (I - K @ H) @ self.P @ (I - K @ H).T + K @ R_cov @ K.T

        return residual


    def inject_error(self, dx):
        self.p += dx[0:3]
        self.v += dx[3:6]
        self.R = self.R @ exp_so3(dx[6:9])
        self.b_a += dx[9:12]
        self.b_g += dx[12:15]