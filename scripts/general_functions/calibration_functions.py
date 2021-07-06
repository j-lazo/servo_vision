import numpy as np
import cv2
import os
from matplotlib import pyplot as plt
from computer_vision import general_functions as gf


def plot_data_equivalence(directory_data):
    dir_imgs = directory_data + 'image_list/'
    dir_sensor_data = directory_data + 'pos_list/'
    dir_joints_data = directory_data + 'joints_list/'

    list_imgs = sorted(os.listdir(dir_imgs))
    list_sensor_data = sorted(os.listdir(dir_sensor_data))
    list_joints_position = sorted(os.listdir(dir_joints_data))


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
    for i, image_name in enumerate(list_imgs[:8]):
        print(i, image_name, 'q1')
        image = cv2.imread(dir_imgs + image_name)
        matrix = read_matrix_file(dir_sensor_data + list_sensor_data[i])
        joints = read_vector_file(dir_joints_data + list_joints_position[i])
        print(joints)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        f, corners = gf.find_corners(gray, pattern_size)
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

    starting_point = 9
    for i, image_name in enumerate(list_imgs[starting_point:]):
        print(i, image_name, 'q2')
        image = cv2.imread(dir_imgs + image_name)
        matrix = read_matrix_file(dir_sensor_data + list_sensor_data[i+starting_point])
        joints = read_vector_file(dir_joints_data + list_joints_position[i+starting_point])
        print(joints)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        f, corners = gf.find_corners(gray, pattern_size)
        point_y = []
        point_x = []

        for x in range(np.shape(corners)[0]):
            point_y.append(corners[x][0][1])
            point_x.append(corners[x][0][0])

        joints_x2.append(int(joints[0]))
        sensors_x.append(float(matrix[1][3]))
        sensors_y.append(float(matrix[0][3]))
        points_y2.append(point_y[7])
        points_x2.append(point_x[7])
        joints_y2.append(int(joints[1]))

    plt.figure()
    plt.subplot(221)
    print(joints_x1, points_x1)
    plt.plot(joints_x1, points_x1, 'ro')
    plt.ylabel('points x (pixels)')
    plt.xlabel('joints x vs points x Q1')
    plt.subplot(222)
    print(joints_y2, points_x2)
    plt.plot(joints_y2, points_x2, 'ro')
    plt.ylabel('points y (pixels)')
    plt.xlabel('joints y vs points Q1')
    plt.subplot(223)
    print(joints_x1, points_y1)
    plt.plot(joints_x1, points_y1, 'ro')
    plt.ylabel('points x (pixels)')
    plt.xlabel('joints vs points Q2')
    plt.subplot(224)
    print(joints_y2, points_y2)
    plt.plot(joints_y2, points_y2, 'ro')
    plt.ylabel('points y (pixels)')
    plt.xlabel('joints vs points Q2')
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
