import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
from matplotlib.gridspec import GridSpec

class MapVisualizer:

    def __init__(self, output_dir='/home/dev/output/test', video_path='/home/dev/output/video.mp4'):
        self.points = []
        self.trajectory = []
        self.ground_truth = []
        self.landmark_counts = []
        self.image_points_green1 = None
        self.image_points_green2 = None
        self.image_points_red = None
        self.image_points_blue = None
        self.rotations = []
        self.harris_points = None
        self.output_dir = output_dir
        self.video_path = video_path
        self.video_writer = None

        # Set up the figure and subplots using GridSpec
        self.fig = plt.figure(figsize=(12, 8), constrained_layout=True)

        # Create a GridSpec with 2 rows and 2 columns
        gs_main = GridSpec(2, 2, height_ratios=[1, 1], width_ratios=[1, 1], figure=self.fig)

        # Left top: Trajectory of last 20 frames with landmarks
        self.ax_recent_trajectory = self.fig.add_subplot(gs_main[0, 0])
        self.ax_recent_trajectory.set_aspect('equal', adjustable='datalim')  # Square aspect

        # Right top: Image with overlaid points
        self.ax_image = self.fig.add_subplot(gs_main[0, 1])

        # Left bottom: "# of landmarks over time"
        self.ax_landmarks = self.fig.add_subplot(gs_main[1, 0])

        # Right bottom: Full trajectory over all frames
        self.ax_full_trajectory = self.fig.add_subplot(gs_main[1, 1])
        self.ax_full_trajectory.set_aspect('equal', adjustable='datalim')  # Square aspect

        # Create output directory if it does not exist
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)


    def add_points(self, points):
        """Add new 3D points to the map.

        Args:
            points (list of tuples): List of (x, y, z) coordinates.
        """
        self.points.extend(points)
        self.landmark_counts.append(len(points))

    def add_pose(self, pose, ground_truth=None):
        """Add a new pose to the trajectory.

        Args:
            pose (tuple): (x, y, z) coordinates of the pose.
        """
        pose = pose.squeeze()
        #R = pose[:3, :3]
        #t = pose[:3, 3]
        #t = -R.T @ t
        #self.trajectory.append((t[0], t[1], t[2]))
        self.trajectory.append((pose[0], pose[1], pose[2]))
        if ground_truth is not None:
            self.ground_truth.append((ground_truth[0], ground_truth[1], ground_truth[2]))
    def update_image(self, image):
        """Update the current image displayed in the plot.

        Args:
            image (ndarray): Grayscale image to display.
        """
        """self.ax_image.cla()
        self.ax_image.set_title("Current Image")
        self.ax_image.imshow(image, cmap='gray')
        self.ax_image.axis('off')
        """
        # Controlla se c'è già un'immagine visualizzata
        if not hasattr(self, 'image_display'):
            # Visualizza l'immagine iniziale
            self.image_display = self.ax_image.imshow(image, cmap='gray')
            self.ax_image.set_title("Current Image")
            self.ax_image.axis('off')
        else:
            # Aggiorna l'immagine senza ridisegnare l'intero asse
            self.image_display.set_data(image)
            for collection in self.ax_image.collections:
                collection.remove()
            for line in self.ax_image.lines:
                line.remove()

        # Mantieni i limiti degli assi fissi
        self.ax_image.set_xlim(0, image.shape[1])
        self.ax_image.set_ylim(image.shape[0], 0)
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
        if (self.harris_points is not None) and (len(self.harris_points) > 1):
            x_vals_harris = [p[0] for p in self.harris_points]
            y_vals_harris = [p[1] for p in self.harris_points]
            self.ax_image.scatter(x_vals_harris, y_vals_harris, c='b', marker='x', s=20)
        
        # Overlay blue crosses on the image
        if (self.image_points_blue is not None) and (len(self.image_points_blue) > 1):
            x_vals_blue = [p[0] for p in self.image_points_blue]
            y_vals_blue = [p[1] for p in self.image_points_blue]
            self.ax_image.scatter(x_vals_blue, y_vals_blue, c='b', marker='x', s=20)
        
    def add_image_points(self, points_green1, points_green2, points_red, harris_points, points_blue):
        """Add points to overlay on the image.

        Args:
            points_green1 (list of tuples): First set of (x, y) coordinates for green crosses.
            points_green2 (list of tuples): Second set of (x, y) coordinates for green crosses.
            points_red (list of tuples): List of (x, y) coordinates for red crosses.
            harris_points (list of tuples): List of (x, y) coordinates for harris points.
            points_blue (list of tuples): List of (x, y) coordinates for blue crosses.
        """
        self.image_points_green1 = points_green1
        self.image_points_green2 = points_green2
        self.image_points_red = points_red
        self.harris_points = harris_points
        self.image_points_blue = points_blue


    def add_pose(self, pose, R=None, ground_truth=None):
        pose = pose.squeeze()
        self.trajectory.append((pose[0], pose[1], pose[2]))
        if R is not None:
            self.rotations.append(R)
        if ground_truth is not None:
            self.ground_truth.append((ground_truth[0], ground_truth[1], ground_truth[2]))

    def euler_from_matrix(self, R):
        sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
        singular = sy < 1e-6

        if not singular:
            x = np.arctan2(R[2, 1], R[2, 2])
            y = np.arctan2(-R[2, 0], sy)
            z = np.arctan2(R[1, 0], R[0, 0])
        else:
            x = np.arctan2(-R[1, 2], R[1, 1])
            y = np.arctan2(-R[2, 0], sy)
            z = 0

        return np.rad2deg((x, y, z))

    def update_plot(self, frame_idx):
        self.ax_recent_trajectory.cla()
        self.ax_recent_trajectory.set_title("Recent Trajectory and Landmarks")
        self.ax_recent_trajectory.set_xlabel('X')
        self.ax_recent_trajectory.set_ylabel('Z')  
        self.ax_recent_trajectory.set_aspect('equal', adjustable='datalim')

        last_n = 20
        trajectory_recent = self.trajectory[-last_n:] if len(self.trajectory) >= last_n else self.trajectory
        rotations_recent = self.rotations[-last_n:] if len(self.rotations) >= last_n else self.rotations
        points_recent = self.points[-last_n:] if len(self.points) >= last_n else self.points
        if points_recent:
            x_vals = [p[0] for p in points_recent]
            z_vals = [p[2] for p in points_recent]  # Use Z values
            self.ax_recent_trajectory.scatter(x_vals, z_vals, c='b', marker='o', s=5)
        if trajectory_recent:
            traj_x = [p[0] for p in trajectory_recent]
            traj_z = [p[2] for p in trajectory_recent]
            self.ax_recent_trajectory.plot(traj_x, traj_z, c='r')

            # Add local frame vectors
            for i, (x, z, R) in enumerate(zip(traj_x, traj_z, rotations_recent)):
                if not np.allclose(R, np.eye(3)):
                    scale_recent_trajectory = 1.5  # Adjust for visualization scale
                    scale_full_trajectory = 0.5  # Adjust for visualization scale
                    R = np.linalg.inv(R)  # Convert from world to local frame
                    vx, vz = R[:3, 0], R[:3, 2]  # Local frame x and z axes

                    # Plot Vx (local x-axis)
                    self.ax_recent_trajectory.arrow(x, z, scale_recent_trajectory * vx[0], scale_recent_trajectory * vx[2], color='g', head_width=scale_recent_trajectory, label='Vx' if i == 0 else "")
                    #plot Vx (local x-axis on the ax_full_trajectory)

                    # Plot Vz (local z-axis)
                    self.ax_recent_trajectory.arrow(x, z, scale_recent_trajectory * vz[0], scale_recent_trajectory * vz[2], color='b', head_width=scale_recent_trajectory, label='Vz' if i == 0 else "")
                   
        #update full trajectory
        self.ax_full_trajectory.cla()
        self.ax_full_trajectory.set_title("Full Trajectory")
        self.ax_full_trajectory.set_xlabel('X')
        self.ax_full_trajectory.set_ylabel('Z')
        self.ax_full_trajectory.set_aspect('equal', adjustable='datalim')
        
        if self.trajectory:
            traj_x_full = [p[0] for p in self.trajectory]
            traj_y_full = [p[2] for p in self.trajectory]
            self.ax_full_trajectory.plot(traj_x_full, traj_y_full, c='r')
        if self.ground_truth:
            gt_x_full = [p[0] for p in self.ground_truth]
            gt_y_full = [p[2] for p in self.ground_truth]
            self.ax_full_trajectory.plot(gt_x_full, gt_y_full, c='black')
        for i, (x, z, R) in enumerate(zip(traj_x_full, traj_y_full, self.rotations)):
            if not np.allclose(R, np.eye(3)):
                scale_full_trajectory = 0.5
                vx = R[:3, 0]  # local x-axis
                vz = R[:3, 2]  # local z-axis

                self.ax_full_trajectory.arrow(
                    x,
                    z,
                    scale_full_trajectory * vx[0],
                    scale_full_trajectory * vx[2],
                    color='g',
                    head_width=scale_full_trajectory,
                    label='Vx' if i == 0 else ""
                )
                self.ax_full_trajectory.arrow(
                    x,
                    z,
                    scale_full_trajectory * vz[0],
                    scale_full_trajectory * vz[2],
                    color='b',
                    head_width=scale_full_trajectory,
                    label='Vz' if i == 0 else ""
                )
        x_center =traj_x[-1]
        y_center =traj_z[-1]
        self.ax_recent_trajectory.set_xlim([x_center - 20, x_center + 20])
        self.ax_recent_trajectory.set_ylim([y_center - 20, y_center + 20])

        



        # Update landmark count plot
        self.ax_landmarks.cla()
        self.ax_landmarks.set_title("# Tracked Landmarks Over Time")
        self.ax_landmarks.set_xlabel('Frame')
        self.ax_landmarks.set_ylabel('# Landmarks')

        if self.landmark_counts:
            self.ax_landmarks.plot(self.landmark_counts, c='g')
        frame_path = os.path.join(self.output_dir, f"frame_{frame_idx:04d}.png")
        self.fig.savefig(frame_path)
        self.fig.savefig(os.path.join(self.output_dir, f"_movie.png"))

        if self.video_writer is None:
            width, height = self.fig.get_size_inches() * self.fig.dpi
            width, height = int(width), int(height)
            self.video_writer = cv2.VideoWriter(self.video_path, cv2.VideoWriter_fourcc(*'mp4v'), 10, (width, height))

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
