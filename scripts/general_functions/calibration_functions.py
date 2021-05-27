import numpy as np
from scipy.linalg import expm, inv
from numpy import dot, eye
import cv2
import os
from matplotlib import pyplot as plt


def plot_data_equivalence(directory_data):
    dir_imgs = directory_data + 'image_list/'
    dir_sensor_data = directory_data + 'pos_list/'
    dir_joints_data = directory_data + 'joints_list/'

    list_imgs = sorted(os.listdir(dir_imgs))
    list_sensor_data = sorted(os.listdir(dir_sensor_data))
    list_joints_position = sorted(os.listdir(dir_joints_data))

    img_points_list = []
    sensor_point_list = []
    joints_point_list = []
    points_x1 = []
    points_y1 = []
    points_x2 = []
    points_y2 = []
    joints_x1 = []
    joints_y1 = []
    joints_x2 = []
    joints_y2 = []
    sensors_x = []
    sensors_y = []

    pattern_size = (7, 5)
    for i, image_name in enumerate(list_imgs[1:7]):
        print(image_name, 'q1')
        image = cv2.imread(dir_imgs + image_name)
        matrix = read_matrix_file(dir_sensor_data + list_sensor_data[i+1])
        joints = read_vector_file(dir_joints_data + list_joints_position[i+1])
        print(joints)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        f, corners = find_corners(gray, pattern_size)
        point_x = []
        point_y = []

        for x in range(np.shape(corners)[0]):
            point_x.append(corners[x][0][0])
            point_y.append(corners[x][0][1])


        joints_x1.append(int(joints[0]))
        sensors_x.append(float(matrix[1][3]))
        sensors_y.append(float(-matrix[0][3]))
        points_y1.append(point_y[7])
        points_x1.append(point_x[7])
        joints_y1.append(int(joints[1]))


    for i, image_name in enumerate(list_imgs[7:]):
        print(image_name, 'q2')
        image = cv2.imread(dir_imgs + image_name)
        matrix = read_matrix_file(dir_sensor_data + list_sensor_data[i+7])
        joints = read_vector_file(dir_joints_data + list_joints_position[i+7])
        print(joints)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        f, corners = find_corners(gray, pattern_size)
        point_y = []
        point_x = []

        for x in range(np.shape(corners)[0]):
            point_y.append(corners[x][0][1])
            point_x.append(corners[x][0][0])

        joints_x2.append(int(joints[0]))
        sensors_x.append(float(matrix[1][3]))
        sensors_y.append(float(-matrix[0][3]))
        points_y2.append(point_y[7])
        points_x2.append(point_x[7])
        joints_y2.append(int(joints[1]))

    plt.figure()
    plt.subplot(221)
    plt.plot(joints_x1, points_x1, 'ro')
    plt.ylabel('points x (pixels)')
    plt.xlabel('joints values')
    plt.subplot(222)
    plt.plot(joints_y2, points_x2, 'ro')
    plt.ylabel('points y (pixels)')
    plt.xlabel('joints values')
    plt.subplot(223)
    plt.plot(joints_x1, points_y1, 'ro')
    plt.ylabel('points x (pixels)')
    plt.xlabel('sensor values')
    plt.subplot(224)
    plt.plot(joints_y2, points_y2, 'ro')
    plt.ylabel('points y (pixels)')
    plt.xlabel('sensor values')
    plt.show()


def read_vector_file(file_path):
    text_file = open(file_path, "r")
    lines = text_file.readlines()
    for i, line in enumerate(lines):

        line = line.replace('[', '')
        line = line.replace(']', '')
        line = line.replace(',', '')

    return line.split()[0], line.split()[1]


def read_matrix_file(file_path):
    matrix = np.zeros((4,4))
    text_file = open(file_path, "r")
    lines = text_file.readlines()
    for i, line in enumerate(lines):
        line = line.replace('[', '')
        line = line.replace(']', '')
        matrix[i][0] = float(line.split()[0])
        matrix[i][1] = float(line.split()[1])
        matrix[i][2] = float(line.split()[2])
        matrix[i][3] = float(line.split()[3])

    return matrix

def find_corners(image, pattern_size):
    """

    @param image: grayscale image
    @param pattern_size:
    @return:
    """
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