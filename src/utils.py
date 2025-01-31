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

def ransac_pnp_pose_estimation(X_3D, P_2D, K):
    # Triangulate the points using the PnP algorithm
    success, R_vec, t, inliers = cv2.solvePnPRansac(objectPoints=X_3D, imagePoints=P_2D, cameraMatrix=K, distCoeffs=None)
    
    if not success or inliers is None:
        # No solution found, return defaults
        print("No solution found in SolvePnPRansac!")
        return X_3D, P_2D, np.eye(3), np.zeros((3, 1)), inliers
    
    R, _ = cv2.Rodrigues(R_vec)
    inliers = inliers.squeeze()
    X_3D_inliers = X_3D[inliers]
    P_2D_inliers = P_2D[inliers]

    return X_3D_inliers, P_2D_inliers, R, t, inliers

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
    C = np.unique(C, axis=1)
    F = np.unique(F, axis=1)
    
    return C, F, Tau

def ComputeCandidates(I, T, ft_params):
    # Compute feature candidates
    C_new = cv2.goodFeaturesToTrack(image=I, mask=None, **ft_params)
    C_new = C_new.squeeze()
    
    # Ensure all feature candidates have the same transformation
    Tau = np.tile(T, (C_new.shape[0], 1, 1))

    # First observations of the track of each keypoint
    F = C_new

    return C_new, F, Tau

def check_for_alpha(S_C, S_F, S_tau, R, K, threshold=0.3):
    """
    Args:
        S_C (np.ndarray): Image points in the current frame (N x 2).
        S_F (np.ndarray): Image points in the first frame (N x 2).
        S_tau (np.ndarray): Camera poses in the first frame (N x 3 x 4).
        R (np.ndarray): Rotation matrix from world coordinates to the current frame (3 x 3).
        K (np.ndarray): Camera intrinsic matrix (3 x 3).
        threshold (float): Threshold for the angle between the two vectors (in radians).
    
    Returns:
        np.ndarray: Mask of the image points that pass the threshold (N x 1).
    """
    # Get number of candidates
    N = S_C.shape[0]

    # Convert image points to normalized image coordinates
    S_C_homogeneous = np.hstack((S_C, np.ones((N, 1))))  # N x 3
    C_camera = (np.linalg.inv(K) @ S_C_homogeneous.T).T  # N x 3

    S_F_homogeneous = np.hstack((S_F, np.ones((N, 1))))  # N x 3
    F_camera = (np.linalg.inv(K) @ S_F_homogeneous.T).T  # N x 3

    # Compute unit-plane normalized image coordinates
    C_camera_normalized = C_camera / np.linalg.norm(C_camera, axis=1, keepdims=True)  # N x 3
    F_camera_normalized = F_camera / np.linalg.norm(F_camera, axis=1, keepdims=True)  # N x 3

    # Transform direction vectors to world coordinates
    C_world = (R.T @ C_camera_normalized.T).T  # N x 3

    F_world = np.zeros_like(F_camera_normalized)
    for i in range(N):
        R_i = S_tau[i, :3, :3]
        F_world[i] = (R_i.T @ F_camera_normalized[i])

    # Compute the angles between the two vectors
    cos_theta = np.sum(C_world * F_world, axis=1)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)  # Ensure values are within valid range
    theta = np.arccos(cos_theta)

    # Check if the angle is greater than the threshold
    mask = theta > threshold
    # print(f"Average alpha (rad): {round(theta.mean(), 3)} | Num passing: {sum(mask)}")

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
    mask = check_for_alpha(S_C=C_cur, 
                            S_F=F_cur,
                            S_tau=tau_cur,
                            R=R,
                            K=K_kitti)

    print("Mask output:")
    print(mask)
    

if __name__ == "__main__":
    main()