#!/usr/bin/env python3
from utils import *
import matplotlib.pyplot as plt
from candidate_kp import track_candidates, triangulate_ransac_pnp, get_new_candidate_points

def main():
    # Initialize FrameManager
    frame_manager = FrameManager(base_path='/home/dev/data', dataset=0, bootstrap_frames=[0, 1])

    # Get images
    img1 = frame_manager.get_previous()
    img2 = frame_manager.get_current()
    frame_manager.update()
    img3 = frame_manager.get_current()

    # Find features in the first image using the Shi-Tomasi method
    feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
    p0 = cv2.goodFeaturesToTrack(img1, mask=None, **feature_params)

    # Parameters for the KLT (Lucas-Kanade) tracker
    lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # Calculate the optical flow from img1 to img2
    p1, st, err = cv2.calcOpticalFlowPyrLK(img1, img2, p0, None, **lk_params)

    # p1 = p1.T
    # p1 = p1.reshape(2, p1.shape[2])

    # Select good points for tracking from 00 to 01
    good_old_01 = p0[st == 1]
    good_new_01 = p1[st == 1]

    # Calculate the optical flow from img2 to img3, continuing the tracking
    p2, st, err = cv2.calcOpticalFlowPyrLK(img2, img3, good_new_01, None, **lk_params)

    good_old_12 = good_new_01[st.flatten() == 1]
    good_new_12 = p2[st.flatten() == 1]

    # Calculate the optical flow directly from img1 to img3
    p2_direct, st_direct, err_direct = cv2.calcOpticalFlowPyrLK(img1, img3, p0, None, **lk_params)

    good_old_direct = p0[st_direct.flatten() == 1]
    good_new_direct = p2_direct[st_direct.flatten() == 1]

    # Create an image to display the results
    output_img = cv2.cvtColor(img3, cv2.COLOR_GRAY2BGR)

    # Draw the tracks from 00 -> 01 -> 02
    for i, (new, old) in enumerate(zip(good_new_12, good_old_12)):
        a, b = int(new.ravel()[0]), int(new.ravel()[1])
        c, d = int(old.ravel()[0]), int(old.ravel()[1])
        cv2.line(output_img, (a, b), (c, d), (255, 0, 0), 2)  # Blue lines for 00 -> 01 -> 02
        cv2.circle(output_img, (a, b), 5, (0, 0, 255), -1)

    # Draw the tracks from 00 -> 02 directly
    for i, (new, old) in enumerate(zip(good_new_direct, good_old_direct)):
        a, b = int(new.ravel()[0]), int(new.ravel()[1])
        c, d = int(old.ravel()[0]), int(old.ravel()[1])
        cv2.line(output_img, (a, b), (c, d), (0, 255, 0), 2)  # Green lines for 00 -> 02 direct
        cv2.circle(output_img, (a, b), 5, (255, 0, 0), -1)

    # Display the image with the tracked points using matplotlib
    plt.imshow(cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB))
    plt.title('KLT Tracking from 00 -> 01 -> 02 and 00 -> 02')
    plt.axis('off')
    plt.show()
    # Prepare points for estimating the fundamental matrix
    # Use the good points tracked directly from 00 -> 02
    good_pts1 = good_old_direct
    good_pts2 = good_new_direct

    # Estimate the fundamental matrix using RANSAC
    F, mask = cv2.findFundamentalMat(good_pts1, good_pts2, cv2.FM_RANSAC)

    # Select inlier points
    inlier_pts1 = good_pts1[mask == 1]
    inlier_pts2 = good_pts2[mask == 1]

    print("Estimated Fundamental Matrix:\n", F)

    #extract the essential matrix
    K_kitti  = np.array([
                [7.18856e+02, 0, 6.071928e+02],
                [0, 7.18856e+02, 1.852157e+02],
                [0, 0, 1]
            ])
    E_essential = K_kitti.T @ F @ K_kitti
    print("Estimated Essential Matrix:\n", E_essential)
    #disambiguate the 4 possible poses

    #recover the rotation and translation from the essential matrix
    _, R, t, _ = cv2.recoverPose(E_essential, inlier_pts1, inlier_pts2, K_kitti)
    print("Recovered Rotation:\n", R)
    print("Recovered Translation:\n", np.round(t,4))
    M = K_kitti @ np.hstack((R, t))

    # Triangulate the points
    points_4D = cv2.triangulatePoints(np.eye(3,4), M, inlier_pts1.T, inlier_pts2.T)
    points_3D = cv2.convertPointsFromHomogeneous(points_4D.T).reshape(-1,3)
    base_path = '/home/dev/data'
    kitti_path = os.path.join(base_path, 'kitti')
    ground_truth = load_ground_truth_kitti(kitti_path)
    # print(np.round(ground_truth[0:4],4))
    
    current_state = {
            "keypoints_2D" : inlier_pts1,
            "keypoints_3D" : points_3D,
            "candidate_2D" : None,
            "candidate_first_2D" : None,
            "candidate_first_camera_pose" : None,
        }
    
    pose_arr = []
    pose_arr.append(np.eye(4)) # Starting position
    pose_arr.append(np.vstack((np.hstack((R, t)),
                               np.array([0,0,0,1]))))

    while frame_manager.has_next():
        print("Next frame")
        frame_manager.update()
        image_current = frame_manager.get_current()
        image_previous = frame_manager.get_previous()
        previous_state = current_state

        P_cur, st, err = cv2.calcOpticalFlowPyrLK(image_previous, image_current, previous_state['keypoints_2D'], None, **lk_params) # TODO: check what lk_params are
        st = st.reshape(-1)
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

            cur_C_to_P_mask = check_for_alpha(C_cur, F_cur, tau_cur, R, t)
            
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


if __name__ == "__main__":
    main()
