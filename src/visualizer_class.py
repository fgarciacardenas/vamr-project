import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
from matplotlib.gridspec import GridSpec

class MapVisualizer:

    def __init__(self, output_dir='output', video_path='output/video.avi'):
        self.points = []
        self.trajectory = []
        self.landmark_counts = []
        self.image_points_green1 = None
        self.image_points_green2 = None
        self.image_points_red = None
        self.output_dir = output_dir
        self.video_path = video_path
        self.video_writer = None

        # Set up the figure and subplots using GridSpec
        self.fig = plt.figure(figsize=(12, 8))

        # Create a GridSpec with 2 rows and 2 columns
        gs_main = GridSpec(2, 2, height_ratios=[3, 1], width_ratios=[1, 1], figure=self.fig)

        # Left top: Trajectory of last 20 frames with landmarks
        self.ax_recent_trajectory = self.fig.add_subplot(gs_main[0, 0])

        # Right top: Image with overlaid points
        self.ax_image = self.fig.add_subplot(gs_main[0, 1])

        # Left bottom: "# of landmarks over time"
        self.ax_landmarks = self.fig.add_subplot(gs_main[1, 0])

        # Right bottom: Full trajectory over all frames
        self.ax_full_trajectory = self.fig.add_subplot(gs_main[1, 1])

        # Create output directory if it does not exist
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def add_points(self, points):
        """Add new 3D points to the map.

        Args:
            points (list of tuples): List of (x, y, z) coordinates.
        """
        self.points.extend(points)
        self.update_landmark_count()

    def add_pose(self, pose):
        """Add a new pose to the trajectory.

        Args:
            pose (tuple): (x, y, z) coordinates of the pose.
        """
        self.trajectory.append((pose[0], pose[1], pose[2]))

    def update_landmark_count(self):
        """Automatically update the landmark count based on the number of points."""
        self.landmark_counts.append(len(self.points))

    def add_image_points(self, points_green1, points_green2, points_red):
        """Add points to overlay on the image.

        Args:
            points_green1 (list of tuples): First set of (x, y) coordinates for green crosses.
            points_green2 (list of tuples): Second set of (x, y) coordinates for green crosses.
            points_red (list of tuples): List of (x, y) coordinates for red crosses.
        """
        self.image_points_green1 = points_green1.squeeze()
        self.image_points_green2 = points_green2.squeeze()
        self.image_points_red = points_red.squeeze()

    def update_image(self, image):
        """Update the current image displayed in the plot.

        Args:
            image (ndarray): Grayscale image to display.
        """
        self.ax_image.cla()
        self.ax_image.set_title("Current Image")
        self.ax_image.imshow(image, cmap='gray')
        self.ax_image.axis('off')

        # Overlay green crosses and lines on the image
        if (self.image_points_green1 is not None) and (self.image_points_green2 is not None) and \
           (len(self.image_points_green1) > 1) and (len(self.image_points_green2) > 1):
            x_vals_green1 = [p[0] for p in self.image_points_green1]
            y_vals_green1 = [p[1] for p in self.image_points_green1]
            x_vals_green2 = [p[0] for p in self.image_points_green2]
            y_vals_green2 = [p[1] for p in self.image_points_green2]
            # Plot green crosses
            self.ax_image.scatter(x_vals_green1, y_vals_green1, c='g', marker='o', s=10)
            self.ax_image.scatter(x_vals_green2, y_vals_green2, c='g', marker='o', s=10)
            # Draw lines connecting corresponding points
            for (x1, y1), (x2, y2) in zip(self.image_points_green1, self.image_points_green2):
                self.ax_image.plot([x1, x2], [y1, y2], c='g', linestyle='-')

        # Overlay red crosses on the image
        if (self.image_points_red is not None) and (len(self.image_points_red) > 1):
            x_vals_red = [p[0] for p in self.image_points_red]
            y_vals_red = [p[1] for p in self.image_points_red]
            self.ax_image.scatter(x_vals_red, y_vals_red, c='r', marker='x', s=20)

    def update_plot(self, frame_idx):
        """Update all subplots with the latest data and save the frame as an image and video."""
        # Update trajectory plot of the last 20 frames with landmarks
        self.ax_recent_trajectory.cla()
        self.ax_recent_trajectory.set_title("Recent Trajectory and Landmarks")
        self.ax_recent_trajectory.set_xlabel('X')
        self.ax_recent_trajectory.set_ylabel('Y')
        self.ax_recent_trajectory.set_aspect('equal')

        # Get the last 20 frames
        last_n = 20
        trajectory_recent = self.trajectory[-last_n:] if len(self.trajectory) >= last_n else self.trajectory
        points_recent = self.points[-last_n:] if len(self.points) >= last_n else self.points

        if points_recent:
            x_vals = [p[0] for p in points_recent]
            y_vals = [p[1] for p in points_recent]
            self.ax_recent_trajectory.scatter(x_vals, y_vals, c='b', marker='o', s=5)

        if trajectory_recent:
            traj_x = [p[0] for p in trajectory_recent]
            traj_y = [p[1] for p in trajectory_recent]
            self.ax_recent_trajectory.plot(traj_x, traj_y, c='r')

        # Update landmark count plot
        self.ax_landmarks.cla()
        self.ax_landmarks.set_title("# Tracked Landmarks Over Time")
        self.ax_landmarks.set_xlabel('Frame')
        self.ax_landmarks.set_ylabel('# Landmarks')

        if self.landmark_counts:
            self.ax_landmarks.plot(self.landmark_counts, c='g')

        # Update full trajectory plot
        self.ax_full_trajectory.cla()
        self.ax_full_trajectory.set_title("Full Trajectory")
        self.ax_full_trajectory.set_xlabel('X')
        self.ax_full_trajectory.set_ylabel('Y')
        self.ax_full_trajectory.set_aspect('equal')

        if self.trajectory:
            traj_x_full = [p[0] for p in self.trajectory]
            traj_y_full = [p[1] for p in self.trajectory]
            self.ax_full_trajectory.plot(traj_x_full, traj_y_full, c='r')

        # Adjust layout
        self.fig.tight_layout()

        # Save the current frame as an image
        frame_path = os.path.join(self.output_dir, f"frame_{frame_idx:04d}.png")
        self.fig.savefig(frame_path)

        # Convert the plot to a video frame
        if self.video_writer is None:
            width, height = self.fig.get_size_inches() * self.fig.dpi
            width, height = int(width), int(height)
            self.video_writer = cv2.VideoWriter(self.video_path, cv2.VideoWriter_fourcc(*'XVID'), 10, (width, height))

        self.fig.canvas.draw()
        frame = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8)
        frame = frame.reshape(int(self.fig.get_figheight() * self.fig.dpi),
                              int(self.fig.get_figwidth() * self.fig.dpi), 3)
        self.video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    def close_video(self):
        """Release the video writer."""
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None

# Example usage
if __name__ == "__main__":
    visualizer = MapVisualizer()

    # Add some points and poses dynamically
    import numpy as np

    for i in range(100):
        # Simulate adding points
        visualizer.add_points([(np.random.uniform(-10, 10),
                                np.random.uniform(-10, 10),
                                np.random.uniform(-10, 10))])

        # Simulate adding a pose
        visualizer.add_pose((i * 0.1, i * 0.1, 0))

        # Simulate updating an image with overlaid points
        dummy_image = np.random.randint(0, 255, (100, 100), dtype=np.uint8)

        # Generate sample data for image points
        num_points = 5
        points_green1 = [(np.random.randint(0, 100), np.random.randint(0, 100)) for _ in range(num_points)]
        points_green2 = [(x + np.random.randint(-5, 5), y + np.random.randint(-5, 5)) for x, y in points_green1]
        points_red = [(np.random.randint(0, 100), np.random.randint(0, 100)) for _ in range(5)]

        visualizer.add_image_points(points_green1, points_green2, points_red)
        visualizer.update_image(dummy_image)

        visualizer.update_plot(i)

    visualizer.close_video()
