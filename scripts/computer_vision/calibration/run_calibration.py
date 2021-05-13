import chessboard
import park_martin
import yaml
from scipy.linalg import expm, inv
from numpy import dot, eye
import numpy as np
import os
import cv2


def hat(v):
    return [[0, -v[2],  v[1]],
            [v[2],     0, -v[0]],
            [-v[1],  v[0],    0]]


def tf_mat(r, t):
    res = eye(4)
    res[0:3, 0:3] = expm(hat(r))
    res[0:3, -1] = t
    return res


def read_text_file(file_path):
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


def main(data_dir):
    corner_list = []
    obj_pose_list = []

    dir_imgs = os.path.join(data_dir, 'image_list')
    dir_poses = os.path.join(data_dir + 'pos_list')

    img_list = sorted(os.listdir(dir_imgs))
    rob_pose_list = sorted(os.listdir(dir_poses))
    camera_matrix, dist_coeffs = chessboard.calibrate_lens(dir_imgs)

    for i, img in enumerate(img_list):
        image = cv2.imread(dir_imgs + img)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        found, corners = chessboard.find_corners(gray)
        corner_list.append(corners)
        if not found:
            raise Exception("Failed to find corners in img # %d" % i)
        rvec, tvec = chessboard.get_object_pose(chessboard.pattern_points, corners, camera_matrix, dist_coeffs)
        object_pose = tf_mat(rvec, tvec)
        obj_pose_list.append(object_pose)

    A, B = [], []
    for i in range(len(rob_pose_list)-1):
        pose_1 = read_text_file(dir_poses + rob_pose_list[i])
        pose_2 = read_text_file(dir_poses + rob_pose_list[i+1])
        p = pose_1, obj_pose_list[i]
        n = pose_2, obj_pose_list[i+1]
        A.append(dot(inv(p[0]), n[0]))
        B.append(dot(inv(p[1]), n[1]))

    # Transformation to chessboard in robot gripper
    X = eye(4)
    Rx, tx = park_martin.calibrate(A, B)
    X[0:3, 0:3] = Rx
    X[0:3, -1] = tx

    print("X: ")
    print(X)

    print("For validation. Printing transformations from the robot base to the camera")
    print("All the transformations should be quite similar")

    for i in range(len(img_list)):
        rob = read_text_file(dir_poses + rob_pose_list[i])
        obj = obj_pose_list[i]
        tmp = dot(rob, dot(X, inv(obj)))
        print(tmp)

    # Here just one is picked but maybe some average can be used instead
    rob = read_text_file(dir_poses + rob_pose_list[0])
    obj = obj_pose_list[0]
    cam_pose = dot(dot(rob, X), inv(obj))

    cam = {'rotation': cam_pose[0:3, 0:3].tolist(),
           'translation': cam_pose[0:3, -1].tolist(),
           'camera_matrix': camera_matrix.tolist(),
           'dist_coeffs': dist_coeffs.tolist()}

    fp = open('camera.yaml', 'w')
    fp.write(yaml.dump(cam))
    fp.close()


if __name__ == "__main__":
    #data_dir = '/home/nearlab/Jorge/current_work/robot_vision/data/calibration/'
    data_dir = os.path.join(os.getcwd(), 'data', 'calibration')
    print(os.getcwd())
    main(data_dir=data_dir)