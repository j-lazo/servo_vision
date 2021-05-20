import serial
from time import sleep
import numpy as np

def store_act_joint_variables(current_joint_variables, act_joint_variables, iterations ):
    act_joint_variables[iterations] = current_joint_variables # should I call another global variable?
    return act_joint_variables


def tf_calculation(act_joint_variables, num_datapoints=21, R=4.5, T=5, l=70):
    # R = 4.5  cables' circle diameter unit in mm
    # T = 5    thread pitch unit in mm
    # l = 70   length of steerable segment unit in mm

    # intialization
    tf_mat = np.zeros((num_datapoints, 4, 4))

    for i in range(num_datapoints):
        phi = np.tan(act_joint_variables[i, 0] / act_joint_variables[i, 1])
        theta = np.sqrt(
            act_joint_variables[i, 0] * act_joint_variables[i, 0] + act_joint_variables[i, 1] * act_joint_variables[
                i, 1]) / R
        # d = act_joint_variables[i, 2] / 2 / np.pi * T  # This number needs to be calibrated.
        d = act_joint_variables[i, 2] /4000.0 * T  # This number needs to be calibrated.
        ## joint variable

        x = l * (1 - np.cos(theta)) * np.cos(phi) / theta
        y = l * (1 - np.cos(theta)) * np.sin(phi) / theta
        z = (l * np.sin(theta) / theta) + d

        tf_mat[i] = np.array([[np.cos(phi) * np.cos(phi) * (np.cos(theta) - 1) + 1,  np.sin(phi) * np.cos(phi) * (np.cos(theta) - 1),      np.cos(phi) * np.sin(theta), x],
                              [np.sin(phi) * np.cos(phi) * (np.cos(theta) - 1),      np.sin(phi) * np.sin(phi) *(np.cos(theta) - 1) + 1,   np.sin(phi) * np.sin(theta), y],
                              [-np.cos(phi) * np.sin(theta),                        -np.sin(phi) * np.sin(theta),                          np.cos(theta),               z],
                              [0, 0, 0, 1]
                              ])

    return tf_mat


def save_pose_txt(tf_mat):

    for i in range(len(tf_mat)):
        if i < 9:
            file_order = '0' + str(i + 1)

        else:
            file_order = str(i + 1)

        write_file_name = 'calibration_00' + file_order + '.txt'
        with open(write_file_name, 'w') as writer:
            writer.write(np.array2string(tf_mat[i]))
    return 0


def main():
    # Example code:
    act = [[44., 50., 0.],
           [188., -80., 0.],
           [-176., 161., 0.],
           [159., 122., 0.],
           [-78., -102., 1.],
           [-105., 73., 1.],
           [-168., 163., 1.],
           [-156., 272., 1.],
           [-59., 190., 1.],
           [-92., 97., 1.],
           [-59., -52., 1.],
           [11., 42., 2.],
           [-50., 55., 2.]]

    tf_max = tf_calculation(act, num_datapoints=13)
    save_pose_txt(tf_max)

    return 0


if __name__ == '__main__':
    main()