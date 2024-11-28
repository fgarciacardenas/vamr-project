import os
import numpy as np
import cv2

def load_ground_truth_kitti(kitti_path):
    poses_path = os.path.join(kitti_path, 'poses', '05.txt')
    ground_truth = np.loadtxt(poses_path)
    ground_truth = ground_truth[:, [-9, -1]]
    return ground_truth

def get_left_images_malaga(malaga_path):
    images_dir = os.path.join(malaga_path, 'malaga-urban-dataset-extract-07_rectified_800x600_Images')
    all_images = sorted(os.listdir(images_dir))
    left_images = all_images[2::2]  # Assuming first two are not images
    return left_images

def load_ground_truth_parking(parking_path):
    poses_path = os.path.join(parking_path, 'poses.txt')
    ground_truth = np.loadtxt(poses_path)
    ground_truth = ground_truth[:, [-9, -1]]
    return ground_truth

def read_image(path, grayscale=True):
    if grayscale:
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    else:
        image = cv2.imread(path, cv2.IMREAD_COLOR)
    
    if image is None:
        raise FileNotFoundError(f"Image not found: {path}")
    return image

def load_matrix(file_path):
    # Load a 3x3 matrix from a text file, handling trailing commas.
    try:
        matrix = np.genfromtxt(file_path, delimiter=',', usecols=(0, 1, 2))
        
        # Verify that the matrix has the correct shape
        if matrix.shape != (3, 3):
            raise ValueError(f"Expected a 3x3 matrix, but got shape {matrix.shape}")
        return matrix
    
    except Exception as e:
        print(f"Error loading K matrix from {file_path}: {e}")
        raise

def main():
    # Define dataset paths
    base_path = '/home/dev/data'
    kitti_path = os.path.join(base_path, 'kitti')
    malaga_path = os.path.join(base_path, 'malaga')
    parking_path = os.path.join(base_path, 'parking')

    # Dataset selection: 0=KITTI, 1=Malaga, 2=Parking
    ds = 1

    # Bootstrap frames
    bootstrap_frames = [0, 1]

    if ds == 0: # KITTI Dataset
        ground_truth = load_ground_truth_kitti(kitti_path)
        last_frame = 4540
        K = np.array([[7.18856e+02,           0, 6.071928e+02],
                      [          0, 7.18856e+02, 1.852157e+02],
                      [          0,           0,            1]
        ])
    elif ds == 1: # Malaga Dataset
        left_images = get_left_images_malaga(malaga_path)
        last_frame = len(left_images)
        K = np.array([[621.18428,         0, 404.00760],
                      [        0, 621.18428, 309.05989],
                      [        0,         0,         1]
        ])
    elif ds == 2: # Parking Dataset
        ground_truth = load_ground_truth_parking(parking_path)
        last_frame = 598
        K = load_matrix(os.path.join(parking_path, 'K.txt'))
    else:
        raise ValueError("Invalid dataset selection. Choose 0 (KITTI), 1 (Malaga), or 2 (Parking).")

    # Bootstrap: Load initial frames
    if ds == 0:
        img0_path = os.path.join(kitti_path, '05', 'image_0', f"{bootstrap_frames[0]:06d}.png")
        img1_path = os.path.join(kitti_path, '05', 'image_0', f"{bootstrap_frames[1]:06d}.png")
        img0 = read_image(img0_path, grayscale=False)
        img1 = read_image(img1_path, grayscale=False)
    elif ds == 1:
        images_dir = os.path.join(malaga_path, 'malaga-urban-dataset-extract-07_rectified_800x600_Images')
        img0_path = os.path.join(images_dir, left_images[bootstrap_frames[0]])
        img1_path = os.path.join(images_dir, left_images[bootstrap_frames[1]])
        img0 = cv2.cvtColor(read_image(img0_path, grayscale=False), cv2.COLOR_BGR2GRAY)
        img1 = cv2.cvtColor(read_image(img1_path, grayscale=False), cv2.COLOR_BGR2GRAY)
    elif ds == 2:
        img0_path = os.path.join(parking_path, 'images', f"img_{bootstrap_frames[0]:05d}.png")
        img1_path = os.path.join(parking_path, 'images', f"img_{bootstrap_frames[1]:05d}.png")
        img0 = cv2.cvtColor(read_image(img0_path, grayscale=False), cv2.COLOR_BGR2GRAY)
        img1 = cv2.cvtColor(read_image(img1_path, grayscale=False), cv2.COLOR_BGR2GRAY)
    else:
        raise ValueError("Invalid dataset selection during bootstrap.")

    # Continuous operation: Process frames from bootstrap_frames[1] + 1 to last_frame
    start_frame = bootstrap_frames[1] + 1
    for i in range(start_frame, last_frame + 1):
        print(f"\n\nProcessing frame {i}\n=====================")

        if ds == 0:
            image_path = os.path.join(kitti_path, '05', 'image_0', f"{i:06d}.png")
            image = read_image(image_path, grayscale=False)
        elif ds == 1:
            if i >= len(left_images):
                print(f"Frame {i} exceeds the number of available images.")
                break
            image_path = os.path.join(
                malaga_path,
                'malaga-urban-dataset-extract-07_rectified_800x600_Images',
                left_images[i]
            )
            image = cv2.cvtColor(read_image(image_path, grayscale=False), cv2.COLOR_BGR2GRAY)
        elif ds == 2:
            image_path = os.path.join(parking_path, 'images', f"img_{i:05d}.png")
            image = cv2.cvtColor(read_image(image_path, grayscale=False), cv2.COLOR_BGR2GRAY)
            image = image.astype(np.uint8)  # Ensure image is in uint8 format
        else:
            raise ValueError("Invalid dataset selection during processing.")

        # Placeholder for processing the image
        # e.g., feature extraction, pose estimation, etc.
        # Example: Display the image (optional)
        cv2.imshow('Current Frame', image)
        cv2.waitKey(1)

    print("Processing completed.")

if __name__ == "__main__":
    main()
