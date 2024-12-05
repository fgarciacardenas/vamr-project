#!/usr/bin/env python3
from utils import *
import matplotlib.pyplot as plt
from candidate_kp import track_candidates

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

    print(p1.shape)
    print(p1)
    p1 = p1.T
    p1 = p1.reshape(2, p1.shape[2])
    print(p1.shape)
    print(p1)
    print(st.shape)
    exit()
    # Select good points for tracking from 00 to 01
    good_old_01 = p0[st.flatten() == 1]
    good_new_01 = p1[st.flatten() == 1]

    print(good_old_01.shape)
    print(good_new_01.shape)
    print(st.shape)
    exit()
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
    inlier_pts1 = good_pts1[mask.ravel() == 1]
    inlier_pts2 = good_pts2[mask.ravel() == 1]

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
    print(np.round(ground_truth[0:4],4))
    
    F_init = inlier_pts1.T
    Tau_init = np.full(F_init.shape[0], np.hstack((R, t)))
    
    C_new, F_new, Tau_new = track_candidates(inlier_pts1.T, F_init, Tau_init, img2, img3)
    print(C_new.shape)
    print(F_new.shape)
    print(Tau_new.shape)
    
    C_prime, F_prime, Tau_prime = expand_C(C_new, F_new, Tau_new, img3, R, t)
    print(C_prime.shape)
    print(F_prime.shape)
    print(Tau_prime.shape)
    # Triangulate the points using the PnP algorithm
    X_new, P_new, R, t = triangulate_ransac_pnp(points_3D, inlier_pts1.T, K_kitti)

if __name__ == "__main__":
    main()\
