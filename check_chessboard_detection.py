# import cv2
# import glob

# # Path to left camera images
# left_images = sorted(glob.glob("images/left/*.*"))

# pattern_size = (9, 6)

# for path in left_images:
#     img = cv2.imread(path)
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     found, corners = cv2.findChessboardCorners(gray, pattern_size)
#     print(f"{path}: {'✅ Found' if found else '❌ Not found'}")

#     if found:
#         cv2.drawChessboardCorners(img, pattern_size, corners, found)
#         cv2.imshow("Corners", img)
#         cv2.waitKey(500)

# cv2.destroyAllWindows()

import cv2

img = cv2.imread("images/left/Im_l_1.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.rotate(gray, cv2.ROTATE_90_CLOCKWISE)

pattern_size = (9, 6)
flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE

found, corners = cv2.findChessboardCorners(gray, pattern_size, flags)
print("Found:", found)

if found:
    cv2.drawChessboardCorners(img, pattern_size, corners, found)

cv2.imshow("Detection", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

