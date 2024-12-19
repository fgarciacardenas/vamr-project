#!/usr/bin/env python3
from utils import *
import numpy as np
import cv2
import matplotlib.pyplot as plt

use_cv2 = True

def initialize_vo(frame_manager, ft_params, klt_params, _debug: bool = False, _gt_init: bool = False):
    
    # ------------ Select frames for initialization ------------
    
    # Select frames
    frame_manager.update()
    I_0 = frame_manager.get_previous() # Frame 1
    I_1 = frame_manager.get_current()  # Frame 2
    frame_manager.update()
    I_2 = frame_manager.get_current()  # Frame 3

    # Prefilter with bilateral filter
    #I_0 = cv2.bilateralFilter(I_0, 5, 5, 20)
    #I_1 = cv2.bilateralFilter(I_1, 5, 5, 20)
    #I_2 = cv2.bilateralFilter(I_2, 5, 5, 20)
    
    # ------------ Establish keypoint correspondences ------------

    # Find features in Frame 1 using Harris
    P_0 = cv2.goodFeaturesToTrack(image=I_0, mask=None, **ft_params)
    # Calculate the optical flow between frame 1 and 2
    P_1, matches_1_2, _ = cv2.calcOpticalFlowPyrLK(prevImg=I_0, nextImg=I_1, prevPts=P_0, nextPts=None, **klt_params)

    # Select good tracking points from frame 1 to 2
    P_1_inliers = P_1[matches_1_2.flatten() == 1]

    # Calculate the optical flow between frame 2 and 3
    P_2, matches_2_3, _ = cv2.calcOpticalFlowPyrLK(prevImg=I_1, nextImg=I_2, prevPts=P_1_inliers, nextPts=None, **klt_params)

    # Select good tracking points from frame 2 to 3
    P_2_inliers = P_2[matches_2_3.flatten() == 1]

    # Select good tracking points from frame 1 to 3
    P_0_inliers = P_0[matches_1_2.flatten() == 1][matches_2_3.flatten() == 1]

    """
    # ------------ Estimate relative pose ------------

    # Estimate the fundamental matrix using RANSAC
    F, mask = cv2.findFundamentalMat(points1=P_0_inliers, points2=P_2_inliers, method=cv2.FM_8POINT)

    # Save outliers
    # P_0_outliers = np.concatenate([
    #     P_0[matches_1_2.flatten() == 0],
    #     P_1[matches_2_3.flatten() == 0],
    #     P_0_inliers[mask.ravel() == 0]
    # ]).reshape([-1,2])
    P_0_outliers = None

    # Select inlier points
    P_0_inliers = P_0_inliers[mask.ravel() == 1]
    P_2_inliers = P_2_inliers[mask.ravel() == 1]

    # Estimate the essential matrix
    K = frame_manager.get_intrinsic_params()
    E = K.T @ F @ K
    """
    P_0_outliers = None
    # Estimate the essential matrix
    K = frame_manager.get_intrinsic_params()
    E, mask = cv2.findEssentialMat(points1=P_0_inliers, points2=P_2_inliers, cameraMatrix=K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    
    # Select inlier points
    P_0_inliers = P_0_inliers[mask.ravel() == 1]
    P_2_inliers = P_2_inliers[mask.ravel() == 1]  
    
    # Recover rotation and translation from the essential matrix
    if use_cv2:
        _, R, t, mask_2, points = cv2.recoverPose(E=E, points1=P_0_inliers, points2=P_2_inliers, cameraMatrix=K, distanceThresh=50.0)
    else:
        P_0_inliers_homogeneous = cv2.convertPointsToHomogeneous(P_0_inliers).squeeze()
        P_2_inliers_homogeneous = cv2.convertPointsToHomogeneous(P_2_inliers).squeeze()
        R, t = decomposeEssentialMatrix(E)
        R, t = disambiguateRelativePose(R, t, P_0_inliers_homogeneous.T, P_2_inliers_homogeneous.T, K, K)
        t = t.reshape(-1,1)
    
    if use_cv2:
        P_0_inliers = P_0_inliers[mask_2.ravel() == 255]
        P_2_inliers = P_2_inliers[mask_2.ravel() == 255]
        
    P_0_inliers = P_0_inliers.squeeze()
    P_2_inliers = P_2_inliers.squeeze()

    M = K @ np.hstack((R, t))

    # Triangulate the points
    if _gt_init:
        # Use ground truth poses to initialize
        pose_0 = frame_manager.get_ground_truth_pose(0)
        pose_2 = frame_manager.get_ground_truth_pose(2)

        # Convert poses to appropriate format
        pose_0[:3, 3] = -pose_0[:3, :3].T @ pose_0[:3, 3]
        pose_2[:3, 3] = -pose_2[:3, :3].T @ pose_2[:3, 3]

        # Compute projection matrices
        projMatrix1 = K @ pose_0[:3]
        projMatrix2 = K @ pose_2[:3]
    else:
        projMatrix1 = K @ np.eye(3,4)
        projMatrix2 = M

    points_4D = cv2.triangulatePoints(
        projMatr1=projMatrix1, 
        projMatr2=projMatrix2, 
        projPoints1=P_0_inliers.T, 
        projPoints2=P_2_inliers.T
    )
    points_3D = cv2.convertPointsFromHomogeneous(src=points_4D.T).squeeze()

    # Print debug outputs
    if _debug:
        print(f"Number of matches (KLT): {np.sum(matches_2_3)} out of {len(P_0)}")
        print(f"Number of matches (Ransac): {len(P_0_inliers)} out of {len(P_0)}")
        #print("Estimated Fundamental Matrix:\n", np.round(F))
        print("Estimated Essential Matrix:\n", np.round(E))
        print("Recovered Rotation (world frame):\n", np.round(R.T,3))
        print("Recovered Translation [X,Y,Z] (world frame):\n", np.round(-R.T@t,3).T[0])
        print("Ground-truth Translation [X,Z] (gt frame):\n", np.round(frame_manager.get_ground_truth()[4],3))
        
        # Plot the inlier correspondences
        plt.figure()
        plt.imshow(I_0)
        plt.scatter(P_0_inliers[:,0], P_0_inliers[:,1], c='r', s=5)
        plt.scatter(P_2_inliers[:,0], P_2_inliers[:,1], c='b', s=5)
        plt.savefig("inlier_correspondences.png")
        
        # Plot the 3D points
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(points_3D[:,0], points_3D[:,1], points_3D[:,2])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.savefig("triangulated_points.png")

    return I_2, P_0_inliers, P_2_inliers, P_0_outliers, points_3D, R, t

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

def disambiguateRelativePose(Rots,u3,points0_h,points1_h,K1,K2):
    """ DISAMBIGUATERELATIVEPOSE- finds the correct relative camera pose (among
     four possible configurations) by returning the one that yields points
     lying in front of the image plane (with positive depth).

     Arguments:
       Rots -  3x3x2: the two possible rotations returned by decomposeEssentialMatrix
       u3   -  a 3x1 vector with the translation information returned by decomposeEssentialMatrix
       p1   -  3xN homogeneous coordinates of point correspondences in image 1
       p2   -  3xN homogeneous coordinates of point correspondences in image 2
       K1   -  3x3 calibration matrix for camera 1
       K2   -  3x3 calibration matrix for camera 2

     Returns:
       R -  3x3 the correct rotation matrix
       T -  3x1 the correct translation vector

       where [R|t] = T_C2_C1 = T_C2_W is a transformation that maps points
       from the world coordinate system (identical to the coordinate system of camera 1)
       to camera 2.
    """
    pass

    # Projection matrix of camera 1
    M1 = K1 @ np.eye(3,4)

    total_points_in_front_best = 0
    for iRot in range(2):
        R_C2_C1_test = Rots[:,:,iRot]
        
        for iSignT in range(2):
            T_C2_C1_test = u3 * (-1)**iSignT
            
            M2 = K2 @ np.c_[R_C2_C1_test, T_C2_C1_test]
            P_C1 = linearTriangulation(points0_h, points1_h, M1, M2)
            
            # project in both cameras
            P_C2 = np.c_[R_C2_C1_test, T_C2_C1_test] @ P_C1
            
            num_points_in_front1 = np.sum(P_C1[2,:] > 0)
            num_points_in_front2 = np.sum(P_C2[2,:] > 0)
            total_points_in_front = num_points_in_front1 + num_points_in_front2
                  
            if (total_points_in_front > total_points_in_front_best):
                # Keep the rotation that gives the highest number of points
                # in front of both cameras
                R = R_C2_C1_test
                T = T_C2_C1_test
                total_points_in_front_best = total_points_in_front

    return R, T


def linearTriangulation(p1, p2, M1, M2):
    """ Linear Triangulation
     Input:
      - p1 np.ndarray(3, N): homogeneous coordinates of points in image 1
      - p2 np.ndarray(3, N): homogeneous coordinates of points in image 2
      - M1 np.ndarray(3, 4): projection matrix corresponding to first image
      - M2 np.ndarray(3, 4): projection matrix corresponding to second image

     Output:
      - P np.ndarray(4, N): homogeneous coordinates of 3-D points
    """
    pass
    assert(p1.shape == p2.shape), "Input points dimension mismatch"
    assert(p1.shape[0] == 3), "Points must have three columns"
    assert(M1.shape == (3,4)), "Matrix M1 must be 3 rows and 4 columns"
    assert(M2.shape == (3,4)), "Matrix M1 must be 3 rows and 4 columns"

    num_points = p1.shape[1]
    P = np.zeros((4, num_points))

    # Linear Algorithm
    for i in range(num_points):
        # Build matrix of linear homogeneous system of equations
        A1 = cross2Matrix(p1[:, i]) @ M1
        A2 = cross2Matrix(p2[:, i]) @ M2
        A = np.r_[A1, A2]

        # Solve the homogeneous system of equations
        _, _, vh = np.linalg.svd(A, full_matrices=False)
        P[:, i] = vh.T[:,-1]

    # Dehomogenize (P is expressed in homoegeneous coordinates)
    P /= P[3,:]

    return P

def cross2Matrix(x):
    """ Antisymmetric matrix corresponding to a 3-vector
     Computes the antisymmetric matrix M corresponding to a 3-vector x such
     that M*y = cross(x,y) for all 3-vectors y.

     Input: 
       - x np.ndarray(3,1) : vector

     Output: 
       - M np.ndarray(3,3) : antisymmetric matrix
    """
    M = np.array([[0,   -x[2], x[1]], 
                  [x[2],  0,  -x[0]],
                  [-x[1], x[0],  0]])
    return M


def decomposeEssentialMatrix(E):
    """ Given an essential matrix, compute the camera motion, i.e.,  R and T such
     that E ~ T_x R
     
     Input:
       - E(3,3) : Essential matrix

     Output:
       - R(3,3,2) : the two possible rotations
       - u3(3,1)   : a vector with the translation information
    """
    pass
    u, _, vh = np.linalg.svd(E)

    # Translation
    u3 = u[:, 2]

    # Rotations
    W = np.array([ [0, -1,  0],
                   [1,  0,  0],
                   [0,  0,  1]])

    R = np.zeros((3,3,2))
    R[:, :, 0] = u @ W @ vh
    R[:, :, 1] = u @ W.T @ vh

    for i in range(2):
        if np.linalg.det(R[:, :, i]) < 0:
            R[:, :, i] *= -1

    if np.linalg.norm(u3) != 0:
        u3 /= np.linalg.norm(u3)

    return R, u3