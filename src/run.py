#!/usr/bin/env python3
from dataloader import *
from initialization import *
from utils import track_candidates
from visualizer_class import MapVisualizer

DATASET = 'parking'
DEBUG = True
GT_INIT = True

"""
Conventions:
- C: Candidate features
- F: First observation of the features
- Tau: Camera pose when the feature was first observed
- P: Inlier features
- X: 3D points in the world frame
- R_a_b: Rotation from frame b to frame a.
- t_a_b: Translation from frame b to frame a.
"""

def main():
    # Initialize FrameManager
    dataset_dir = {'kitti': 0, 'malaga': 1, 'parking': 2}
    frame_manager = FrameManager(base_path='/home/dev/data', dataset=dataset_dir[DATASET], bootstrap_frames=[0, 1])
    
    # Load K matrix
    K = frame_manager.K
    
    # Configure modules
    
    ft_params = dict(maxCorners=100, qualityLevel=0.001, minDistance=20, blockSize=3, k=0.04, useHarrisDetector=True)
    klt_params = dict(winSize=(21, 21), maxLevel=4, criteria=(cv2.TERM_CRITERIA_COUNT + cv2.TERM_CRITERIA_EPS, 30, 0.001))
    angle_thr = np.deg2rad(2)

    # Initialize position
    I_2, P_0_inliers, P_2_inliers, P_0_outliers, X_2, cam_R, cam_t = initialize_vo(
        frame_manager=frame_manager, 
        ft_params=ft_params, 
        klt_params=klt_params, 
        _debug=DEBUG,
        _gt_init = GT_INIT
    )

    # Initialize current state dictionary
    current_state = {
        "keypoints_2D" : P_2_inliers,
        "keypoints_3D" : X_2,
        "candidate_2D" : None,
        "candidate_first_2D" : None,
        "candidate_first_camera_pose" : None,
    }

    # Get starting position
    pose_arr = [np.eye(4)]
    pose_arr.append(np.vstack((np.hstack((cam_R, cam_t)),
                                np.array([0,0,0,1]))))
    
    # Initialize visualizer
    visualizer = MapVisualizer()
    visualizer.add_points(X_2)
    visualizer.add_pose(-cam_R.T@cam_t)
    visualizer.add_image_points(P_0_inliers, P_2_inliers, P_0_outliers, P_0_inliers)
    visualizer.update_image(I_2)
    
    iFrame = 0
    while frame_manager.has_next():
        # Update frame manager
        print(f"Frame {iFrame}")
        frame_manager.update()

        # Obtain current and previous frames
        I_curr = frame_manager.get_current()
        I_prev = frame_manager.get_previous()

        # Update state
        previous_state = current_state

        # Compute optical flow for inlier features
        P, matches, _ = cv2.calcOpticalFlowPyrLK(
            prevImg=I_prev, 
            nextImg=I_curr, 
            prevPts=previous_state['keypoints_2D'], 
            nextPts=None, 
            **klt_params
        )
        
        # Flatten matches for ease of use
        matches = matches.flatten()

        # Select good tracking points from previous and new frame
        P_1_klt = P[matches == 1]

        # Update previous state
        X = previous_state['keypoints_3D'][matches == 1]

        # Compute 3D-2D correspondences
        X, P_1_inliers, R, t, rs_inliers = ransac_pnp_pose_estimation(
            X_3D=X, 
            P_2D=P_1_klt, 
            K=K
        )

        # Compute KLT visualization
        P_0_inliers = previous_state['keypoints_2D'][matches == 1]
        P_0_outliers = np.concatenate([previous_state['keypoints_2D'][matches == 0], P_0_inliers[np.setdiff1d(np.arange(len(P_0_inliers)), rs_inliers)]])
        P_0_inliers = P_0_inliers[rs_inliers]

        next_pose = np.vstack((np.hstack((R, t)),
                               np.array([0,0,0,1])))
        pose_arr.append(next_pose)

        # Compute feature candidates
        C_candidate, F_candidate, Tau_candidate = ComputeCandidates(
            I=I_curr, 
            T=next_pose, 
            ft_params=ft_params
        )

        # Generate feature tracks
        if previous_state["candidate_2D"] is not None:
            
            # Compute optical flow for tracked features
            S_C, matches, _ = cv2.calcOpticalFlowPyrLK(
                prevImg=I_prev, 
                nextImg=I_curr, 
                prevPts=previous_state['candidate_2D'], 
                nextPts=None, 
                **klt_params
            )

            # Flatten matches for ease of use
            matches = matches.flatten()

            # Find inliers in the previous state
            S_C = S_C[matches == 1].squeeze()
            S_F = previous_state['candidate_first_2D'][matches == 1].squeeze()
            S_tau = previous_state['candidate_first_camera_pose'][matches == 1]
            
            # Angle between tracked features for thresholding
            cur_C_to_P_mask = check_for_alpha(S_C=S_C, 
                                              S_F=S_F,
                                              S_tau=S_tau,
                                              R=next_pose[:3,:3],
                                              K=K,
                                              threshold=angle_thr)

            # Add inlier feature candidates not already in P
            member_and_alpha_mask = np.zeros(S_C.shape[0])
            for i, C in enumerate(S_C):
                if not cur_C_to_P_mask[i]:
                    continue
                is_member = np.any(np.all(np.isclose(P_1_inliers, C, rtol=5e-3, atol=1e-1), axis=1))
                if not is_member:
                    member_and_alpha_mask[i] = 1
            if sum(member_and_alpha_mask):
                C_inlier = S_C[member_and_alpha_mask == 1]
                S_F_inlier = S_F[member_and_alpha_mask == 1]
                S_tau_inlier = S_tau[member_and_alpha_mask == 1]

                # Triangulate inlier points
                points_4D = []
                for can, first,tau in zip(C_inlier, S_F_inlier, S_tau_inlier):
                    points_4D.append(cv2.triangulatePoints(
                        projMatr1=K @ tau[:3],
                        projMatr2=K @ next_pose[:3], 
                        projPoints1=first, 
                        projPoints2=can
                    ))
                points_4D = np.array(points_4D).T
                points_3D = cv2.convertPointsFromHomogeneous(src=points_4D.T).squeeze()
                # Update P and X using valid candidates
                P_1_inliers = np.vstack([P_1_inliers, C_inlier])
                X = np.vstack([X, points_3D])

            # Track point that did not pass the alpha threshold
            S_C = S_C[member_and_alpha_mask == 0]
            S_F = S_F[member_and_alpha_mask == 0]
            S_tau = S_tau[member_and_alpha_mask == 0]

            # Append feature candidates
            c_duplicate_mask = []
            for i, new_C in enumerate(C_candidate):
                is_member = np.any(np.all(np.isclose(new_C, S_C, rtol=5e-3, atol=1e-1), axis=1))
                if not is_member:
                    c_duplicate_mask.append(i)
            S_C = np.vstack([S_C, C_candidate[c_duplicate_mask]])
            S_F = np.vstack([S_F, F_candidate[c_duplicate_mask]])
            S_tau = np.vstack([S_tau, Tau_candidate[c_duplicate_mask]])

        else:
            S_C, S_F, S_tau = C_candidate, F_candidate, Tau_candidate

        current_state = {
            "keypoints_2D" : P_1_inliers,
            "keypoints_3D" : X,
            "candidate_2D" : S_C,
            "candidate_first_2D" : S_F,
            "candidate_first_camera_pose" : S_tau,
        }
        # Update visualizer
        visualizer.add_points(X)
        visualizer.add_pose(-next_pose[:3,:3].T@next_pose[:3,3],R = R, ground_truth=frame_manager.get_current_ground_truth())
        visualizer.add_image_points(P_0_inliers, P_1_inliers, P_0_outliers, C_candidate)
        visualizer.update_image(I_curr)
        visualizer.update_plot(iFrame)
        iFrame += 1
        if iFrame >= 60:
            break

    visualizer.close_video()
    print(f"Video saved at {visualizer.video_path}")

if __name__ == "__main__":
    main()
