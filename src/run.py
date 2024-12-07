#!/usr/bin/env python3
from dataloader import *
from initialization import *
from utils import track_candidates
from visualizer_class import MapVisualizer

DATASET = 'kitty'

def main():
    # Initialize FrameManager
    dataset_dir = {'kitty': 0, 'malaga': 1, 'parking': 2}
    frame_manager = FrameManager(base_path='../data', dataset=dataset_dir[DATASET], bootstrap_frames=[0, 1])
    lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    
    # Load K matrix
    K_kitti = frame_manager.K
    
    # Configure modules
    ft_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7, useHarrisDetector=False)
    klt_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # Initialize position
    I_2, P_0_inliers, P_2_inliers, P_0_outliers, X_2, cam_R, cam_t = initialize_vo(frame_manager, ft_params, klt_params, _debug=True)
    exit(1)

    current_state = {
        "keypoints_2D" : P_2_inliers,
        "keypoints_3D" : X_2,
        "candidate_2D" : None,
        "candidate_first_2D" : None,
        "candidate_first_camera_pose" : None,
    }

    pose_arr = []
    pose_arr.append(np.eye(4)) # Starting position
    pose_arr.append(np.vstack((np.hstack((cam_R, cam_t)),
                                   np.array([0,0,0,1]))))
    # Initialize visualizer
    visualizer = MapVisualizer()
    visualizer.add_points(X_2)
    visualizer.add_pose(cam_t)
    visualizer.add_image_points(P_0_inliers, P_2_inliers, P_0_outliers)
    visualizer.update_image(I_2)
    iFrame = 0
    while frame_manager.has_next():
        print("Next frame")
        frame_manager.update()
        image_current = frame_manager.get_current()
        image_previous = frame_manager.get_previous()
        previous_state = current_state

        P_cur, st, err = cv2.calcOpticalFlowPyrLK(image_previous, image_current, previous_state['keypoints_2D'], None, **klt_params) # TODO: check what klt_params are
        st = st.reshape(-1)
        P_cur = P_cur.reshape(-1, 2)
        P_cur = P_cur[st == 1] # TODO: see if we can do this step in KLT ^
        X_cur = previous_state['keypoints_3D'][st == 1]

        X_cur, P_cur, R, t = triangulate_ransac_pnp(X_cur, P_cur, K_kitti)

        new_C_S, new_C_F, new_C_tau = get_new_candidate_points(image_current, R, t)

        if previous_state["candidate_2D"] is not None:
            C_cur, st, err = cv2.calcOpticalFlowPyrLK(image_previous, image_current, previous_state['candidate_2D'], None, **lk_params) # TODO: check what lk_params are
            st = st.reshape(-1)
            C_cur = C_cur[st == 1] # TODO: see if we can do this step in KLT ^
            F_cur = previous_state['candidate_first_2D'][st == 1]
            tau_cur = previous_state['candidate_first_camera_pose'][st == 1]

            cur_C_to_P_mask = check_for_alpha(C_cur, F_cur, tau_cur, R, t, K_kitti)

            P_cur += C_cur[cur_C_to_P_mask] # TODO: Find correct concat operation with uniqueness check
            C_cur = C_cur[cur_C_to_P_mask == 0] # TODO: Find right way to invert mask

            C_cur = C_cur + new_C_S # TODO: These should all be concat operations
            F_cur = F_cur + new_C_F
            tau_cur = tau_cur + new_C_tau

        else:
            C_cur, F_cur, tau_cur = new_C_S, new_C_F, new_C_tau

        current_state = {
            "keypoints_2D" : P_cur,
            "keypoints_3D" : X_cur,
            "candidate_2D" : C_cur,
            "candidate_first_2D" : F_cur,
            "candidate_first_camera_pose" : tau_cur,
        }

        previous_pose = pose_arr[-1]
        current_transformation = np.vstack((np.hstack((R, t)),
                                            np.array([0,0,0,1])))
        next_pose = current_transformation@previous_pose
        pose_arr.append(next_pose)
    

        visualizer.add_points(X_cur)
        visualizer.add_pose(pose_arr[-1][:3,3])
        # visualizer.add_image_points(P_0_inliers, P_2_inliers, P_0_outliers) TODO: ??
        visualizer.update_image(image_current)

        visualizer.update_plot(iFrame)
        iFrame += 1

    visualizer.close_video()
    print(f"Video saved at {visualizer.video_path}")

if __name__ == "__main__":
    main()
