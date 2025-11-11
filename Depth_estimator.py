"""
stereo_depth_estimation.py
--------------------------
Compute disparity and approximate depth using stereo images.
Before running:
  pip install opencv-python numpy
"""

import cv2
import numpy as np

left_image_path = "left.jpg"
right_image_path = "right.jpg"

# need to calibrate the stereo camera first to get the calculation parameters
# Z = (f * B) / d
# Where:
# Z = depth (distance to object)
# f = focal length of the camera
# B = baseline (distance between cameras)
# d = disparity (difference in pixel coordinates of the same point in both images)