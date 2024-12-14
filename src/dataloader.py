#!/usr/bin/env python3
import os
import numpy as np
import cv2

def load_ground_truth_kitti(kitti_path):
    poses_path = os.path.join(kitti_path, 'poses', '05.txt')
    ground_truth = np.loadtxt(poses_path)
    ground_truth = ground_truth[:, [-9,-5,-1]] # [tx,ty,tz]
    return ground_truth

def get_left_images_malaga(malaga_path):
    images_dir = os.path.join(malaga_path, 'malaga-urban-dataset-extract-07_rectified_800x600_Images')
    all_images = sorted(os.listdir(images_dir))
    left_images = all_images[2::2]
    return left_images

def load_ground_truth_parking(parking_path):
    poses_path = os.path.join(parking_path, 'poses.txt')
    ground_truth = np.loadtxt(poses_path)
    ground_truth = ground_truth[:, [-9,-5,-1]] # [tx,ty,tz]
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

class FrameManager:
    """
    A class to manage image frames from different datasets.

    Attributes:
        base_path (str): Base directory path where datasets are stored.
        dataset (int): Dataset selection (0=KITTI, 1=Malaga, 2=Parking).
        last_frame (int): The last frame index for the selected dataset.
        K (np.ndarray): Intrinsic camera matrix for the selected dataset.
        current_index (int): Current frame index.
        previous_image (np.ndarray): Previous image frame.
        current_image (np.ndarray): Current image frame.
        left_images (list): List of left images (for Malaga dataset).
        dataset_specific_data (dict): Additional data specific to the dataset.
    """

    def __init__(self, base_path, dataset=0, bootstrap_frames=[0, 1]):
        """
        Initializes the FrameManager with the specified dataset.

        Args:
            base_path (str): Base directory path where datasets are stored.
            dataset (int, optional): Dataset selection (0=KITTI, 1=Malaga, 2=Parking). Defaults to 0.
            bootstrap_frames (list, optional): Initial frames to load. Defaults to [0, 1].
        """
        self.base_path = base_path
        self.dataset = dataset
        self.bootstrap_frames = bootstrap_frames
        self.previous_image = None
        self.current_image = None
        self.ground_truth = None
        self.left_images = []
        self.dataset_specific_data = {}

        # Initialize dataset-specific parameters
        self._initialize_dataset()

        # Load bootstrap frames
        self.current_index = self.bootstrap_frames[1]
        self.previous_image, self.current_image = self._load_bootstrap_images()

    def _initialize_dataset(self):
        """Initializes dataset-specific paths, ground truth, and camera matrix."""
        if self.dataset == 0:  # KITTI Dataset
            kitti_path = os.path.join(self.base_path, 'kitti')
            self.dataset_specific_data['kitti_path'] = kitti_path
            self.ground_truth = load_ground_truth_kitti(kitti_path)
            self.last_frame = 4540
            self.K = np.array([[7.18856e+02,           0, 6.071928e+02],
                               [          0, 7.18856e+02, 1.852157e+02],
                               [          0,           0,            1]])
        elif self.dataset == 1:  # Malaga Dataset
            malaga_path = os.path.join(self.base_path, 'malaga-urban-dataset-extract-07')
            self.dataset_specific_data['malaga_path'] = malaga_path
            self.left_images = get_left_images_malaga(malaga_path)
            self.last_frame = len(self.left_images) - 1
            self.K = np.array([[621.18428,         0, 404.00760],
                               [        0, 621.18428, 309.05989],
                               [        0,         0,         1]])
        elif self.dataset == 2:  # Parking Dataset
            parking_path = os.path.join(self.base_path, 'parking')
            self.dataset_specific_data['parking_path'] = parking_path
            self.ground_truth = load_ground_truth_parking(parking_path)
            self.last_frame = 598
            K_path = os.path.join(parking_path, 'K.txt')
            self.K = load_matrix(K_path)
        else:
            raise ValueError("Invalid dataset selection. Choose 0 (KITTI), 1 (Malaga), or 2 (Parking).")

    def _load_bootstrap_images(self):
        """Loads the initial bootstrap images based on the dataset."""
        if self.dataset == 0:  # KITTI Dataset
            kitti_path = self.dataset_specific_data['kitti_path']
            img0_path = os.path.join(kitti_path, '05', 'image_0', f"{self.bootstrap_frames[0]:06d}.png")
            img1_path = os.path.join(kitti_path, '05', 'image_0', f"{self.bootstrap_frames[1]:06d}.png")
            img0 = read_image(img0_path, grayscale=True)
            img1 = read_image(img1_path, grayscale=True)
        elif self.dataset == 1:  # Malaga Dataset
            malaga_path = self.dataset_specific_data['malaga_path']
            images_dir = os.path.join(malaga_path, 'malaga-urban-dataset-extract-07_rectified_800x600_Images')
            img0_path = os.path.join(images_dir, self.left_images[self.bootstrap_frames[0]])
            img1_path = os.path.join(images_dir, self.left_images[self.bootstrap_frames[1]])
            img0 = read_image(img0_path, grayscale=True)
            img1 = read_image(img1_path, grayscale=True)
        elif self.dataset == 2:  # Parking Dataset
            parking_path = self.dataset_specific_data['parking_path']
            img0_path = os.path.join(parking_path, 'images', f"img_{self.bootstrap_frames[0]:05d}.png")
            img1_path = os.path.join(parking_path, 'images', f"img_{self.bootstrap_frames[1]:05d}.png")
            img0 = read_image(img0_path, grayscale=True)
            img1 = read_image(img1_path, grayscale=True)
            img0 = img0.astype(np.uint8)
            img1 = img1.astype(np.uint8)
        else:
            raise ValueError("Invalid dataset selection during bootstrap.")
        
        return img0, img1
    
    def get_intrinsic_params(self):
        return self.K
    
    def get_ground_truth(self):
        if self.ground_truth is None:
            print("No ground truth available!")
            return []    
        return self.ground_truth
    
    def get_current_ground_truth(self):
        if self.ground_truth is None:
            return np.array([0.0, 0.0, 0.0])
        return self.ground_truth[self.current_index]

    def get_current(self):
        """
        Returns the current image frame.

        Returns:
            np.ndarray: Current image.
        """
        return self.current_image

    def get_previous(self):
        """
        Returns the previous image frame.

        Returns:
            np.ndarray: Previous image.
        """
        return self.previous_image

    def update(self):
        """
        Updates the frame indices and loads the next image.

        Raises:
            IndexError: If the next frame exceeds the last frame.
        """
        next_index = self.current_index + 1
        if next_index > self.last_frame:
            raise IndexError("Reached the end of the dataset.")

        # Load the next image based on the dataset
        if self.dataset == 0:  # KITTI Dataset
            kitti_path = self.dataset_specific_data['kitti_path']
            image_path = os.path.join(kitti_path, '05', 'image_0', f"{next_index:06d}.png")
            new_image = read_image(image_path, grayscale=True)
        elif self.dataset == 1:  # Malaga Dataset
            if next_index >= len(self.left_images):
                raise IndexError(f"Frame {next_index} exceeds the number of available images.")
            malaga_path = self.dataset_specific_data['malaga_path']
            images_dir = os.path.join(malaga_path, 'malaga-urban-dataset-extract-07_rectified_800x600_Images')
            image_path = os.path.join(images_dir, self.left_images[next_index])
            new_image = read_image(image_path, grayscale=True)
        elif self.dataset == 2:  # Parking Dataset
            parking_path = self.dataset_specific_data['parking_path']
            image_path = os.path.join(parking_path, 'images', f"img_{next_index:05d}.png")
            new_image = read_image(image_path, grayscale=True)
            new_image = new_image.astype(np.uint8)
        else:
            raise ValueError("Invalid dataset selection during processing.")

        # Update previous and current images
        self.previous_image = self.current_image
        self.current_image = new_image
        self.current_index = next_index

    def has_next(self):
        """
        Checks if there are more frames to process.

        Returns:
            bool: True if more frames are available, False otherwise.
        """
        return self.current_index < self.last_frame
