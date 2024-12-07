#!/usr/bin/env python3
import cv2
import numpy as np
from dataloader import *

def track_candidates(C_old, F_old, Tau_old, img1, img2):
    # Parameters for the KLT (Lucas-Kanade) tracker
    lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    
    # Calculate the optical flow from img1 to img2
    C_new, st, err = cv2.calcOpticalFlowPyrLK(img1, img2, C_old, None, **lk_params)

    C_new = C_new.T
    C_new.reshape(2, C_new.shape[2])
    st = st.flatten()
    # Select good points for tracking from 00 to 01
    good_old_01 = C_old[:][st == 1]
    good_new_01 = C_new[:][st == 1]
    F_old = F_old[:][st == 1]
    Tau_old = Tau_old[:][st == 1]
    return good_new_01, F_old, Tau_old

def triangulate_ransac_pnp(X_old, P_new, K):
    # Triangulate the points using the PnP algorithm
    _, R_vec, t, inliers = cv2.solvePnPRansac(X_old, P_new, K, None)
    R, _ = cv2.Rodrigues(R_vec)
    inliers = inliers.reshape(-1)
    X_old = X_old[inliers]
    P_new = P_new[inliers]
    return X_old, P_new, R, t

def expand_C(C, F, Tau, img, R, t):
    # Extract Harris corners from the image
    C_new = cv2.goodFeaturesToTrack(img, maxCorners=200, qualityLevel=0.01, minDistance=7, blockSize=7)
    
    # Add new non overlapping corners to the existing corners
    C = np.concatenate((C, C_new), axis=1)
    F = np.concatenate((F, C_new), axis=1)
    
    # Add the pose to the new ones
    T = np.concatenate((R, t), axis=1)
    Tau = np.concatenate((Tau, np.full(C_new.shape[1], T)), axis=1)
    
    # Remove duplicates
    C = np.unique(C, axis=1) #TODO: maybe there is a better way to do this)
    F = np.unique(F, axis=1)
    
    return C, F, Tau

def get_new_candidate_points(image, R, t):
    C_new = cv2.goodFeaturesToTrack(image, maxCorners=200, qualityLevel=0.01, minDistance=7, blockSize=7)
    T = np.concatenate((R, t), axis=1)
    Tau = np.tile(T, (C_new.shape[0], 1, 1))
    F = C_new

    return C_new, F, Tau

def check_for_alpha(C_cur, F_cur, tau_cur, R, t, K, threshold=20):
    N = C_cur.shape[0]

    # Convert image points to normalized camera coordinates
    C_cur_homogeneous = np.hstack((C_cur, np.ones((N, 1))))  # N x 3
    C_camera = (np.linalg.inv(K) @ C_cur_homogeneous.T).T  # N x 3

    F_cur_homogeneous = np.hstack((F_cur, np.ones((N, 1))))  # N x 3
    F_camera = (np.linalg.inv(K) @ F_cur_homogeneous.T).T  # N x 3

    # Normalize the direction vectors
    C_camera_normalized = C_camera / np.linalg.norm(C_camera, axis=1, keepdims=True)  # N x 3
    F_camera_normalized = F_camera / np.linalg.norm(F_camera, axis=1, keepdims=True)  # N x 3

    # Transform direction vectors to world coordinates
    C_world = (R.T @ C_camera_normalized.T).T  # N x 3

    F_world = np.zeros_like(F_camera_normalized)
    for i in range(N):
        R_i = tau_cur[i, :3, :3]
        F_world[i] = (R_i.T @ F_camera_normalized[i])

    # Compute the angles between the two vectors
    cos_theta = np.sum(C_world * F_world, axis=1)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)  # Ensure values are within valid range
    theta = np.arccos(cos_theta)

    # Check if the angle is greater than the threshold
    mask = theta > np.radians(threshold)

    return mask

def main():
    import numpy as np

    # Sample camera intrinsic matrix K_kitti
    K_kitti = np.array([
        [718.8560, 0, 607.1928],
        [0, 718.8560, 185.2157],
        [0, 0, 1]
    ])

    # Rotation matrix R and translation vector t for C_cur
    R = np.eye(3)  # Identity rotation
    t = np.zeros(3)  # Zero translation

    # Sample image points C_cur and F_cur (N x 2 arrays)
    C_cur = np.array([
        [100, 100],
        [150, 200],
        [200, 250],
        [250, 300],
        [300, 350]
    ])
    F_cur = np.array([
        [102, 98],
        [148, 202],
        [198, 252],
        [248, 298],
        [298, 352]
    ])

    N = C_cur.shape[0]

    # Sample tau_cur (N x 3 x 4 array)
    tau_cur = np.zeros((N, 3, 4))
    for i in range(N):
        # Rotation and translation for each F_cur
        angle = np.radians(5 * (i+1))  # Varying rotation
        R_i = np.array([
            [ np.cos(angle), 0, np.sin(angle)],
            [            0, 1,            0],
            [-np.sin(angle), 0, np.cos(angle)]
        ])
        t_i = np.array([0.1 * (i+1), 0, 0])  # Varying translation

        tau_cur[i, :3, :3] = R_i
        tau_cur[i, :3, 3] = t_i

    # Call the check_for_alpha function
    mask = check_for_alpha(C_cur, F_cur, tau_cur, R, t, K_kitti)

    print("Mask output:")
    print(mask)
    

if __name__ == "__main__":
    main()