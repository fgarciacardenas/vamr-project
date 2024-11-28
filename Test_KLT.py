import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the images from the 'data/' folder
img1 = cv2.imread('data/kitti/05/image_0/000000.png', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('data/kitti/05/image_0/000002.png', cv2.IMREAD_GRAYSCALE)

# Check if images are loaded properly
if img1 is None or img2 is None:
    print("Error: Image not found in the 'data/' folder. Make sure the images are present.")
    exit()

# Find features in the first image using the Shi-Tomasi method
feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
p0 = cv2.goodFeaturesToTrack(img1, mask=None, **feature_params)

# Parameters for the KLT (Lucas-Kanade) tracker
lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Calculate the optical flow between img1 and img2
p1, st, err = cv2.calcOpticalFlowPyrLK(img1, img2, p0, None, **lk_params)

# Check if any points are found
if p1 is None or st is None:
    print("Error: No points were found for tracking.")
    exit()

# Select good points for tracking
good_new = p1[st == 1]
good_old = p0[st == 1]

# Create an image to display the results
output_img = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)

# Draw the tracks of the points
for i, (new, old) in enumerate(zip(good_new, good_old)):
    a, b = int(new[0]), int(new[1])
    c, d = int(old[0]), int(old[1])
    cv2.line(output_img, (a, b), (c, d), (0, 255, 0), 2)
    cv2.circle(output_img, (a, b), 5, (0, 0, 255), -1)

# Display the image with the tracked points using matplotlib
plt.imshow(cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB))
plt.title('KLT Tracking')
plt.axis('off')
plt.show()

# Save the image with the tracked points
cv2.imwrite('klt_tracking_output.png', output_img)
