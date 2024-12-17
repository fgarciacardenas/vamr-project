#!/usr/bin/env python3
import cv2
import numpy as np
from dataloader import FrameManager
from visualizer_class import MapVisualizer
from utils import ComputeCandidates
import os

def main():
    # Specify dataset and frames to test
    DATASET = 'parking'  # or 'kitti', 'malaga'
    FRAME_IDX_1 = 1  # First frame index
    FRAME_IDX_2 = 5  # Second frame index

    # Initialize FrameManager to load images and ground truth poses
    dataset_dir = {'kitti': 0, 'malaga': 1, 'parking': 2}
    frame_manager = FrameManager(base_path='/home/dev/data', dataset=dataset_dir[DATASET], bootstrap_frames=[FRAME_IDX_1, FRAME_IDX_2])

    # Load camera intrinsic matrix
    K = frame_manager.get_intrinsic_params()

    # Load two images and their ground truth poses
    I_0 = frame_manager.get_frame(FRAME_IDX_1)
    I_1 = frame_manager.get_frame(FRAME_IDX_2)
    pose_0 = frame_manager.get_ground_truth_pose(FRAME_IDX_1)
    pose_1 = frame_manager.get_ground_truth_pose(FRAME_IDX_2)
    
    pose_0[:3, 3] = -pose_0[:3, :3].T @ pose_0[:3, 3]
    pose_1[:3, 3] = -pose_1[:3, :3].T @ pose_1[:3, 3] 
    
    # Extract features from the first image
    ft_params = dict(maxCorners=100, qualityLevel=0.01, minDistance=7, blockSize=7)
    C_0, F_0, Tau_0 = ComputeCandidates(I=I_0, T=pose_0, ft_params=ft_params)

    # Track features from the first image to the second image using optical flow
    klt_params = dict(winSize=(21, 21), maxLevel=4,
                      criteria=(cv2.TERM_CRITERIA_COUNT + cv2.TERM_CRITERIA_EPS, 30, 0.01))
    C_1, st, err = cv2.calcOpticalFlowPyrLK(I_0, I_1, C_0, None, **klt_params)
    st = st.flatten()

    # Select good points
    C_0_good = C_0[st == 1]
    C_1_good = C_1[st == 1]
    Tau_0_good = Tau_0[st == 1]

    # Triangulate 3D points using the camera poses and matched features
    points_4D = cv2.triangulatePoints(projMatr1=K @ pose_0[:3],
                                      projMatr2=K @ pose_1[:3],
                                      projPoints1=C_0_good.T,
                                      projPoints2=C_1_good.T)
    points_3D = cv2.convertPointsFromHomogeneous(points_4D.T).reshape(-1, 3)

    # Visualize the two poses and the point cloud
    visualizer = MapVisualizer()
    visualizer.add_points(points_3D)
    visualizer.add_pose(-pose_0[:3,:3].T @ pose_0[:3,3])
    visualizer.add_pose(-pose_1[:3,:3].T @ pose_1[:3,3])
    visualizer.add_image_points(C_0_good, C_1_good, None)
    visualizer.update_image(I_1)
    visualizer.update_plot(frame_idx=0)

    # Save visualization
    output_dir = visualizer.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    frame_path = os.path.join(output_dir, f"triangulation_test.png")
    visualizer.fig.savefig(frame_path)
    print(f"Visualization saved at {frame_path}")

if __name__ == "__main__":
    main()
