from stereo_calibration import StereoCalibration

input_path = "images"                # main folder containing
chessboard_size = (9, 6)           # inner corners per chessboard row and column
square_size = 0.025                # in meters

# Step 1
stereo_calibrator = StereoCalibration(input_path, chessboard_size, square_size)

# Step 2
stereo_calibrator.create_chessboard_points()

# Step 3
stereo_calibrator.calibrate()

# Step 4: Save calibration to XML
stereo_calibrator.save_stereo_calibration()
