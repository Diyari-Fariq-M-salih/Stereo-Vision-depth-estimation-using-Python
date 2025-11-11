from typing import List, Tuple
import cv2
import numpy as np
import os
from tqdm import tqdm


class StereoCalibration:
    """Stereo calibration class adapted for your folder layout.
    Expects:
        ./images/left/*.png
        ./images/right/*.png
    """

    def __init__(self, input_path: str, chessboard_size: Tuple[int, int], square_size: float) -> None:
        self.input_path = input_path               # e.g., "images"
        self.chessboard_size = chessboard_size     # (cols, rows)
        self.square_size = square_size             # in meters
        self.corners_path = os.path.join(input_path, "corners")
        self._initialize()
        self._init_paths()

    def _init_paths(self) -> None:
        """Initialize paths to left and right image folders."""
        self._left_image_path = os.path.join(self.input_path, "left")
        self._right_image_path = os.path.join(self.input_path, "right")

        # Accept .png, .jpg, .jpeg
        valid_exts = (".png", ".jpg", ".jpeg")
        self._left_image_names = sorted(
            [f for f in os.listdir(self._left_image_path) if f.lower().endswith(valid_exts)]
        )
        self._right_image_names = sorted(
            [f for f in os.listdir(self._right_image_path) if f.lower().endswith(valid_exts)]
        )

        self._left_image_paths = [os.path.join(self._left_image_path, f) for f in self._left_image_names]
        self._right_image_paths = [os.path.join(self._right_image_path, f) for f in self._right_image_names]

    def _initialize(self) -> None:
        self._objpoints = []   # 3D real-world points
        self._imgpoints_left = []  # 2D left image points
        self._imgpoints_right = []  # 2D right image points

        self._stereo_map_left = None
        self._stereo_map_right = None
        self._image_size = None

    def create_3d_chessboard_points(self) -> np.ndarray:
        objp = np.zeros((np.prod(self.chessboard_size), 3), np.float32)
        objp[:, :2] = np.indices(self.chessboard_size).T.reshape(-1, 2)
        objp *= self.square_size
        return objp

    def create_2d_chessboard_points(self, image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        found, corners = cv2.findChessboardCorners(gray, self.chessboard_size, flags)
        if not found:
            return None
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        return corners

    def create_chessboard_points(self) -> None:
        """Find chessboard corners for all stereo pairs."""
        print("[INFO] Searching for chessboard corners...")
        print(f"[DEBUG] Left path: {self._left_image_path}")
        print(f"[DEBUG] Found {len(self._left_image_paths)} left images, {len(self._right_image_paths)} right images")
        print(f"[DEBUG] Example left image: {self._left_image_paths[0] if self._left_image_paths else 'None'}")

        objp = self.create_3d_chessboard_points()
        success = 0

        for left_path, right_path in tqdm(zip(self._left_image_paths, self._right_image_paths),
                                          total=len(self._left_image_paths)):
            left = cv2.imread(left_path)
            right = cv2.imread(right_path)

            left_corners = self.create_2d_chessboard_points(left)
            right_corners = self.create_2d_chessboard_points(right)

            if left_corners is not None and right_corners is not None:
                self._objpoints.append(objp)
                self._imgpoints_left.append(left_corners)
                self._imgpoints_right.append(right_corners)
                success += 1

        self._image_size = (left.shape[1], left.shape[0])
        print(f"[INFO] Chessboard detected in {success}/{len(self._left_image_paths)} stereo pairs.")

    def calibrate(self):
        """Calibrate both cameras and compute rectification."""
        print("[INFO] Calibrating left camera...")
        _, mtx_left, dist_left, _, _ = cv2.calibrateCamera(
            self._objpoints, self._imgpoints_left, self._image_size, None, None
        )

        print("[INFO] Calibrating right camera...")
        _, mtx_right, dist_right, _, _ = cv2.calibrateCamera(
            self._objpoints, self._imgpoints_right, self._image_size, None, None
        )

        print("[INFO] Running stereo calibration...")
        flags = cv2.CALIB_FIX_INTRINSIC
        criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-5)
        _, mtx_left, dist_left, mtx_right, dist_right, R, T, E, F = cv2.stereoCalibrate(
            self._objpoints,
            self._imgpoints_left,
            self._imgpoints_right,
            mtx_left,
            dist_left,
            mtx_right,
            dist_right,
            self._image_size,
            criteria=criteria,
            flags=flags,
        )

        print(f"[INFO] Baseline (T): {np.linalg.norm(T):.4f} m")

        print("[INFO] Rectifying stereo pair...")
        R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
            mtx_left, dist_left, mtx_right, dist_right, self._image_size, R, T, alpha=0
        )

        print("[INFO] Creating stereo maps...")
        self._stereo_map_left = cv2.initUndistortRectifyMap(
            mtx_left, dist_left, R1, P1, self._image_size, cv2.CV_16SC2
        )
        self._stereo_map_right = cv2.initUndistortRectifyMap(
            mtx_right, dist_right, R2, P2, self._image_size, cv2.CV_16SC2
        )
        print("[INFO] Calibration completed successfully!")

    def save_stereo_calibration(self):
        """Save maps to XML."""
        print("[INFO] Saving stereo calibration maps...")
        xml_path = os.path.join(self.input_path, "stereo_calibration.xml")
        fs = cv2.FileStorage(xml_path, cv2.FILE_STORAGE_WRITE)
        fs.write("stereo_map_left_x", self._stereo_map_left[0])
        fs.write("stereo_map_left_y", self._stereo_map_left[1])
        fs.write("stereo_map_right_x", self._stereo_map_right[0])
        fs.write("stereo_map_right_y", self._stereo_map_right[1])
        fs.release()
        print(f"[INFO] Saved calibration maps to: {xml_path}")

    def rectify_calibration_images(self):
        """Rectify all calibration images."""
        print("[INFO] Rectifying all calibration images...")
        out_left = os.path.join(self.input_path, "rectified", "left")
        out_right = os.path.join(self.input_path, "rectified", "right")
        os.makedirs(out_left, exist_ok=True)
        os.makedirs(out_right, exist_ok=True)

        for i, (l, r) in enumerate(zip(self._left_image_paths, self._right_image_paths)):
            imgL = cv2.imread(l)
            imgR = cv2.imread(r)
            rectL = cv2.remap(imgL, self._stereo_map_left[0], self._stereo_map_left[1], cv2.INTER_LINEAR)
            rectR = cv2.remap(imgR, self._stereo_map_right[0], self._stereo_map_right[1], cv2.INTER_LINEAR)
            cv2.imwrite(os.path.join(out_left, f"rectified_{i:02d}.png"), rectL)
            cv2.imwrite(os.path.join(out_right, f"rectified_{i:02d}.png"), rectR)

        print(f"[INFO] Rectified images saved under {self.input_path}/rectified/")
