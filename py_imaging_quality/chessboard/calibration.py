"""Camera calibration using chessboard patterns, equivalent to MATLAB chessboard/figure_CaliProcess.m."""
import os
import glob
import numpy as np
import cv2
import tkinter as tk
from tkinter import filedialog, simpledialog
import matplotlib.pyplot as plt


def calibrate_camera(image_dir=None, square_size_mm=30):
    """Calibrate camera using chessboard images.

    Args:
        image_dir: directory containing chessboard images. If None, user selects.
        square_size_mm: chessboard square size in mm.

    Returns:
        params: dict with calibration parameters
    """
    if image_dir is None:
        root = tk.Tk()
        root.withdraw()
        image_dir = filedialog.askdirectory(title='Select image folder')
        root.destroy()
        if not image_dir:
            return None

    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff']:
        image_files.extend(glob.glob(os.path.join(image_dir, ext)))

    if not image_files:
        print('No images found in directory')
        return None

    print(f'Loading {len(image_files)} images...')
    print('Detecting checkerboard points...')

    # Chessboard pattern size
    pattern_size = (9, 6)

    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
    objp *= square_size_mm

    objpoints = []
    imgpoints = []
    img_size = None

    for fname in image_files:
        img = cv2.imread(fname)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if img_size is None:
            img_size = gray.shape[::-1]

        ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)
        if ret:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
            imgpoints.append(corners2)

    print(f'Found chessboard in {len(objpoints)}/{len(image_files)} images')
    print('Calibrating camera...')

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, img_size, None, None)

    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        mean_error += error
    mean_error /= len(objpoints)

    params = {
        'camera_matrix': mtx,
        'dist_coeffs': dist,
        'image_size': img_size,
        'focal_length': (mtx[0, 0], mtx[1, 1]),
        'principal_point': (mtx[0, 2], mtx[1, 2]),
        'radial_distortion': (dist[0, 0], dist[0, 1], dist[0, 4]),
        'tangential_distortion': (dist[0, 2], dist[0, 3]),
        'mean_reprojection_error': mean_error,
        'num_images_used': len(objpoints)
    }

    print(f'Calibration complete.')
    print(f'Focal length: {params["focal_length"]}')
    print(f'Radial distortion: {params["radial_distortion"]}')
    print(f'Mean reprojection error: {mean_error:.4f} px')

    return params


def undistort_image(img, params):
    """Undistort an image using calibration parameters.

    Args:
        img: input image
        params: calibration parameter dict from calibrate_camera

    Returns:
        undistorted image
    """
    h, w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(
        params['camera_matrix'], params['dist_coeffs'], (w, h), 1, (w, h))
    dst = cv2.undistort(img, params['camera_matrix'], params['dist_coeffs'],
                        None, newcameramtx)
    return dst
