# """
# stereo_depth_estimation.py
# --------------------------
# Compute disparity and approximate depth using stereo images.
# Before running:
#   pip install opencv-python numpy tqdm
# """

# # need to calibrate the stereo camera first to get the calculation parameters
# # Z = (f * B) / d
# # Where:
# # Z = depth (distance to object)
# # f = focal length of the camera
# # B = baseline (distance between cameras)
# # d = disparity (difference in pixel coordinates of the same point in both images)

# # calibration done using Stereo_calibration_Driver.py, now we can resume depth estimation

import cv2
import numpy as np

calibration_file = "images/stereo_calibration.xml"
left_image_path = "images/left/Im_L_1.png"
right_image_path = "images/right/Im_R_1.png"

# Baseline
baseline = 0.1582

# Load rectification maps from calibration
fs = cv2.FileStorage(calibration_file, cv2.FILE_STORAGE_READ)
left_map_x = fs.getNode("stereo_map_left_x").mat()
left_map_y = fs.getNode("stereo_map_left_y").mat()
right_map_x = fs.getNode("stereo_map_right_x").mat()
right_map_y = fs.getNode("stereo_map_right_y").mat()
fs.release()

# === STEP 1: Load and rectify images ===
left_img = cv2.imread(left_image_path, cv2.IMREAD_GRAYSCALE)
right_img = cv2.imread(right_image_path, cv2.IMREAD_GRAYSCALE)

if left_img is None or right_img is None:
    raise IOError("Error loading left/right images.")

# Rectify
left_rect = cv2.remap(left_img, left_map_x, left_map_y, cv2.INTER_LINEAR)
right_rect = cv2.remap(right_img, right_map_x, right_map_y, cv2.INTER_LINEAR)

# === STEP 2: Compute disparity map ===
# Create StereoSGBM matcher
window_size = 5
min_disp = 0
num_disp = 16 * 5  # must be multiple of 16
stereo = cv2.StereoSGBM_create(
    minDisparity=min_disp,
    numDisparities=num_disp,
    blockSize=window_size,
    P1=8 * 3 * window_size ** 2,
    P2=32 * 3 * window_size ** 2,
    disp12MaxDiff=1,
    uniquenessRatio=10,
    speckleWindowSize=100,
    speckleRange=32
)

print("[INFO] Computing disparity map...")
disparity = stereo.compute(left_rect, right_rect).astype(np.float32) / 16.0

# Normalize for display
disp_vis = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

# === STEP 3: Convert disparity → depth ===
focal_length = 700

depth_map = np.zeros_like(disparity, dtype=np.float32)
depth_map[disparity > 0] = (focal_length * baseline) / disparity[disparity > 0]

# === STEP 4: Show results ===
cv2.imshow("Left Rectified", left_rect)
cv2.imshow("Right Rectified", right_rect)
cv2.imshow("Disparity Map", disp_vis)

# Print distance at center pixel
h, w = depth_map.shape
center_depth = depth_map[h // 2, w // 2]
print(f"Estimated distance at image center: {center_depth:.3f} meters")

cv2.waitKey(0)
cv2.destroyAllWindows()
