from stereo_calibration import StereoCalibration

# === USER SETTINGS ===
input_path = "images"         # Matches your actual folder structure
chessboard_size = (11, 7)      # Adjust to your real board (10x7 squares → (9,6))
square_size = 0.025           # 25 mm squares

# === CALIBRATION PIPELINE ===
print("[INFO] Initializing stereo calibration...")
calib = StereoCalibration(input_path, chessboard_size, square_size)

print("[INFO] Detecting chessboard corners...")
calib.create_chessboard_points()

print("[INFO] Running calibration...")
calib.calibrate()

print("[INFO] Saving calibration maps...")
calib.save_stereo_calibration()

print("[INFO] Rectifying calibration images...")
calib.rectify_calibration_images()

print("\n✅ Calibration complete!")
