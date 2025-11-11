# Stereo_calibration_Driver.py
from stereo_calibration import StereoCalibration

# === USER SETTINGS ===
input_path = "."              # Path to the folder containing /left and /right
chessboard_size = (8, 5)           # Inner corners per row and column
square_size = 0.025                # In meters (25 mm)

# === STEP 1
stereo_calibrator = StereoCalibration(
    input_path=input_path,
    chessboard_size=chessboard_size,
    square_size=square_size
)

# === STEP 2
print("\n[INFO] Creating chessboard points...")
stereo_calibrator.create_chessboard_points()

# === STEP 3
print("\n[INFO] Running stereo calibration...")
stereo_calibrator.calibrate()

# === STEP 4
print("\n[INFO] Saving calibration data...")
stereo_calibrator.save_stereo_calibration()

# === STEP 
print("\n[INFO] Rectifying calibration images...")
stereo_calibrator.rectify_calibration_images()

print("\n✅ Stereo calibration complete. Results saved to:")
print("  → images/params/stereo_calibration.xml")
print("  → images/rectified/")
