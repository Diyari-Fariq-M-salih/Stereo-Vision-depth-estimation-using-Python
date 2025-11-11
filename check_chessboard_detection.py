import cv2
import numpy as np

# Path to one of your calibration images
img_path = "images/left/Im_l_1.png"
pattern_size = (9, 6)  # inner corners

# Load image
img = cv2.imread(img_path)
if img is None:
    raise FileNotFoundError(f"Image not found: {img_path}")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.equalizeHist(gray)  # enhance contrast

flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK

found = False
for i, rot in enumerate([0, cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE]):
    if rot != 0:
        rotated = cv2.rotate(gray, rot)
    else:
        rotated = gray.copy()

    f, corners = cv2.findChessboardCorners(rotated, pattern_size, flags)
    print(f"Rotation {i}: Found={f}")

    if f:
        found = True
        # visualize detection
        vis = cv2.cvtColor(rotated, cv2.COLOR_GRAY2BGR)
        cv2.drawChessboardCorners(vis, pattern_size, corners, f)
        cv2.imshow("Detected Chessboard", vis)
        cv2.waitKey(0)
        break

if not found:
    print("❌ Chessboard still not detected. Try adjusting brightness or using a smaller pattern size (e.g. (8,5)).")

cv2.destroyAllWindows()


