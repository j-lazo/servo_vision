import numpy as np
from scipy.linalg import expm, inv
from numpy import dot, eye
import cv2
import os


def find_corners(image, pattern_size):

    found, corners = cv2.findChessboardCorners(image, pattern_size)
    term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.1)
    if found:
        cv2.cornerSubPix(image, corners, (5, 5), (-1, -1), term)
    return found, corners


def draw_corners(image, corners, pattern_size):
    color_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    cv2.drawChessboardCorners(color_image, pattern_size, corners, True)
    return color_image


def get_object_pose(object_points, image_points, camera_matrix, dist_coeffs):
    ret, rvec, tvec = cv2.solvePnP(object_points, image_points, camera_matrix, dist_coeffs)
    return rvec.flatten(), tvec.flatten()


def calibrate_lens(dir_imgs, pattern_size, square_size):
    image_list = sorted(os.listdir(dir_imgs))
    img_points, obj_points = [], []
    h,w = 0, 0
    pattern_points = np.zeros((np.prod(pattern_size), 3), np.float32)
    pattern_points[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)
    pattern_points *= square_size

    for image_name in image_list:

        img = cv2.imread(dir_imgs + image_name)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = img.shape[:2]
        found, corners = find_corners(img, pattern_size)

        if not found:
            raise Exception("chessboard calibrate_lens Failed to find corners in img")
        img_points.append(corners.reshape(-1, 2))
        obj_points.append(pattern_points)

    camera_matrix = np.zeros((3,3))
    dist_coeffs = np.zeros(5)
    #rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, (w,h))
    cv2.calibrateCamera(obj_points, img_points, (w, h), camera_matrix, dist_coeffs)
    return camera_matrix, dist_coeffs


def hat(v):
    return [[0,    -v[2],  v[1]],
            [v[2],     0, -v[0]],
            [-v[1],  v[0],    0]]


def tf_mat(r, t):
    res = eye(4)
    res[0:3, 0:3] = expm(hat(r))
    res[0:3, -1] = t
    return res