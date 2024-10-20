import numpy as np


def prepare_matrix(uv, xyz):
    # mat_leftは11列で、各点につき2行必要
    mat_left = np.zeros((2 * len(xyz), 11))
    # mat_rightは各点につき2要素
    mat_right = np.zeros(2 * len(xyz))

    for i in range(len(xyz)):
        u = uv[i][0]
        v = uv[i][1]
        x, y, z = xyz[i]

        # 1行目を割り当てる
        mat_left[2*i] = [x, y, z, 1, 0, 0, 0, 0, -u*x, -u*y, -u*z]
        # 2行目を割り当てる
        mat_left[2*i+1] = [0, 0, 0, 0, x, y, z, 1, -v*x, -v*y, -v*z]

        # mat_rightの要素を割り当てる
        mat_right[2*i] = u
        mat_right[2*i+1] = v

    p, residuals, rank, s = np.linalg.lstsq(mat_left, mat_right, rcond=None)
    return p

def p_vector(p):
    p1 = np.array([p[0], p[1], p[2]])
    p2 = np.array([p[4], p[5], p[6]])
    p3 = np.array([p[8], p[9], p[10]])
    p14 = p[3]
    p24 = p[7]
    return p1, p2, p3, p14, p24

def pose_recon_2c(cam_num, P, POSE):
    result = [] 
    for i in range(len(POSE[1])):
        mat_l = np.empty((0, 3))
        mat_r = np.empty((0, 1))
        for j in range(cam_num):
            p1, p2, p3, p14, p24 = p_vector(P[j])
            u = POSE[j][i][0] * p3 - p1
            v = POSE[j][i][1] * p3 - p2
            mat_l = np.vstack((mat_l, u))
            mat_l = np.vstack((mat_l, v))
            mat_r = np.vstack((mat_r, np.array([p14 - POSE[j][i][0]])))
            mat_r = np.vstack((mat_r, np.array([p24 - POSE[j][i][1]])))

        mat_l_pinv = np.linalg.pinv(mat_l)
        glo = np.dot(mat_l_pinv, mat_r)
        result.append(glo.ravel().tolist())
    return np.array(result)


def MPJPE(predicted_points: np.ndarray, true_points: np.ndarray) -> float:
    # Ensure that 'predicted_points' and 'true_points' are numpy arrays with the shape (n_points, 3)
    assert predicted_points.shape == true_points.shape, "Shapes must match"

    # Calculate the Euclidean distance for each corresponding pair of points
    distances = np.linalg.norm(predicted_points - true_points, axis=1)

    # Compute the average of these distances
    mpjpe = np.mean(distances)
    return mpjpe

def sigmoid(x: np.ndarray, k: int = 50) -> np.ndarray:
    weights = 1 / (1 + np.exp(-k * (x - np.mean(x))))
    normalized_weights = weights / np.sum(weights)
    return normalized_weights