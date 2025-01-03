#!/usr/bin/env python3
from utils import *

def initialize_vo(frame_manager, ft_params, klt_params, _debug: bool = False):
    
    # ------------ Select frames for initialization ------------
    
    # Select frames
    I_0 = frame_manager.get_previous() # Frame 1
    I_1 = frame_manager.get_current()  # Frame 2
    frame_manager.update()
    I_2 = frame_manager.get_current()  # Frame 3
    
    # ------------ Establish keypoint correspondences ------------

    # Find features in Frame 1 using Shi-Tomasi
    P_0 = cv2.goodFeaturesToTrack(I_0, mask=None, **ft_params)

    # Calculate the optical flow between frame 1 and 2
    P_1, matches_1_2, _ = cv2.calcOpticalFlowPyrLK(I_0, I_1, P_0, None, **klt_params)

    # Select good tracking points from frame 1 to 2
    P_1_inliers = P_1[matches_1_2.flatten() == 1]

    # Calculate the optical flow between frame 2 and 3
    P_2, matches_2_3, _ = cv2.calcOpticalFlowPyrLK(I_1, I_2, P_1_inliers, None, **klt_params)

    # Select good tracking points from frame 2 to 3
    P_2_inliers = P_2[matches_2_3.flatten() == 1]

    # Select good tracking points from frame 1 to 3
    P_0_inliers = P_0[matches_1_2.flatten() == 1][matches_2_3.flatten() == 1]

    # ------------ Estimate relative pose ------------

    # Estimate the fundamental matrix using RANSAC
    F, mask = cv2.findFundamentalMat(P_0_inliers, P_2_inliers, cv2.FM_RANSAC)

    # Select inlier points
    P_0_inliers = P_0_inliers[mask.ravel() == 1]
    P_2_inliers = P_2_inliers[mask.ravel() == 1]

    # Estimate the essential matrix
    K = frame_manager.get_intrinsic_params()
    E = K.T @ F @ K

    # Recover rotation and translation from the essential matrix
    _, R, t, _ = cv2.recoverPose(E, P_0_inliers, P_2_inliers, K)
    M = K @ np.hstack((R, t))

    # Triangulate the points
    points_4D = cv2.triangulatePoints(np.eye(3,4), M, P_0_inliers.T, P_2_inliers.T)
    points_3D = cv2.convertPointsFromHomogeneous(points_4D.T).squeeze()

    # Print debug outputs
    if _debug:
        print("Estimated Fundamental Matrix:\n", F)
        print("Estimated Essential Matrix:\n", E)
        print("Recovered Rotation (camera frame):\n", R)
        print("Recovered Translation (camera frame):\n", np.round(t,4))
        print("Recovered Translation (camera frame):\n", print(np.round(frame_manager.get_ground_truth()[0:4],4)))

    return I_0, I_1, I_2, P_0_inliers, P_2_inliers

    # TODO: Populate C and Tau
    # Shape P_i as [2,N]
    #P_0 = P_0.squeeze().T
    #P_2 = P_2.squeeze().T
    #print(inlier_pts1.shape)
    #print(inlier_pts2.shape)
    # print(inlier_pts1.T.shape)

    # F_init = inlier_pts1.T
    # Tau_init = np.full(F_init.shape[0], np.hstack((R, t)))
    
    # C_new, F_new, Tau_new = track_candidates(inlier_pts1.T, F_init, Tau_init, I_1, I_2)
    # print(C_new.shape)
    # print(F_new.shape)
    # print(Tau_new.shape)
    
    # C_prime, F_prime, Tau_prime = expand_C(C_new, F_new, Tau_new, I_2, R, t)
    # print(C_prime.shape)
    # print(F_prime.shape)
    # print(Tau_prime.shape)
    # # Triangulate the points using the PnP algorithm
    # X_new, P_new, R, t = triangulate_ransac_pnp(points_3D, inlier_pts1.T, K)
