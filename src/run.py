#!/usr/bin/env python3
from dataloader import *
from initialization import *
from utils import track_candidates

DATASET = 'kitty'

def main():
    # Initialize FrameManager
    dataset_dir = {'kitty': 0, 'malaga': 1, 'parking': 2}
    frame_manager = FrameManager(base_path='/home/dev/data', dataset=dataset_dir[DATASET], bootstrap_frames=[0, 1])

    # Configure modules
    ft_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
    klt_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # Initialize position
    initialize_vo(frame_manager, ft_params, klt_params, _debug = True)

if __name__ == "__main__":
    main()
