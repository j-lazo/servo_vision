import numpy as np
import cv2
from general_functions import calibration_functions as cf
import argparse
import os
from em_tracking import sample_test as emt
from sksurgerynditracker.nditracker import NDITracker
from control import pySerial_request as pysr
import computer_vision.calibration.run_calibration as cal
from control import mega_communication as mc
from control import find_arduino
from time import sleep
from computer_vision.calibration import detect_circle
from control import general_control_functions as gcf
import random
import tensorflow as tf
from computer_vision import general_functions as cvf
from computer_vision import call_models as cm
import datetime
import time
from general_functions import data_managament as dm


def run_experiment(type_experiment, user_id):

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    date_experiment = datetime.datetime.now()

    results_folder = ''.join([os.getcwd(), '/results/experiment_', type_experiment, '_',
                              'usr_', user_id, '_', date_experiment.strftime("%d_%m_%Y_%H_%M"), '/'])

    if not os.path.isdir(results_folder):
        os.mkdir(results_folder)

    name_video = results_folder + 'experiment_lumen_output_video' + type_experiment + '_' + \
                 'usr_' + user_id + '_' + date_experiment.strftime("%d_%m_%Y_%H_%M")

    out = cv2.VideoWriter(name_video + '_.avi', fourcc, 20.0, (300, 300))
    cap = cv2.VideoCapture(0)
    port_arduino = find_arduino.find_arduino()
    print('Arduino detected at:', port_arduino)
    arduino_port_1 = mc.serial_initialization(arduino_com_port_1=str(port_arduino))
    initial_z = mc.serial_request(arduino_port_1)[2]/800
    current_act_z = mc.serial_request(arduino_port_1)[2]/800

    max_speed = 35
    mins_speed = 1
    prev_key = 0
    defined_speed = 5
    z = 0
    # define arrays to save the variables to analyze
    center_points_x = []
    center_points_y = []
    filtered_points_x = []
    filtered_points_y = []

    project_folder = os.getcwd() + '/scripts/computer_vision/models/weights/'
    folder_model = 'Transpose_ResUnet_lr_0.001_bs_16_rgb_27_05_2021_13_03'
    # load the model and get input size of the model
    model, input_size = cm.load_model(project_folder, folder_model)
    print('input size', input_size)
    print('Please input 6 points for each corner of the trajectory to follow')
    while cap.isOpened():
        key = cv2.waitKey(1) & 0xFF
        prev = 0
        ret, frame = cap.read()

        if key == 43 and defined_speed <= max_speed:
            defined_speed = defined_speed + 1
            print('speed', defined_speed)
            key = prev_key
        if key == 45 and defined_speed >= mins_speed:
            defined_speed = defined_speed - 1
            print('speed', defined_speed)
        if key == 83:
            print('right')
            mc.serial_actuate(-defined_speed, 0, z, arduino_port_1)
        elif key == 81:
            print('left')
            mc.serial_actuate(defined_speed, 0, z, arduino_port_1)
        elif key == 82:
            print('up')
            mc.serial_actuate(0, defined_speed, z, arduino_port_1)
        elif key == 84:
            print('down')
            mc.serial_actuate(0, -defined_speed, z, arduino_port_1)
        elif key == ord('f'):
            print('forward')
            z = z + 1
            mc.serial_actuate(0, 0, z, arduino_port_1)
        elif key == ord('b'):
            print('backwards')
            z = z - 1
            mc.serial_actuate(0, 0, z, arduino_port_1)

        elif key == ord('s'):
            print('stop')
            z_stop = mc.serial_request(arduino_port_1)[2] / 800
            z = mc.serial_request(arduino_port_1)[2] / 800
            mc.serial_actuate(0, 0, z_stop, arduino_port_1)
        # else:
        # mc.serial_actuate(0, 0, z, arduino_port_1)

        prev_key = key

        if ret is True:

            output_image, point_x, point_y = cm.detect_lumen(model, frame)
            center_points_x.append(point_x)
            center_points_y.append(point_y)

            ptx = dm.calculate_average_points(center_points_x[-4:])
            pty = dm.calculate_average_points(center_points_y[-4:])
            filtered_points_x.append(ptx)
            filtered_points_y.append(pty)
            # if a point is detected
            current_act_joint_variable = mc.serial_request(arduino_port_1)
            if not (np.isnan(ptx)) and not (np.isnan(pty)):
                h, w, d = np.shape(output_image)
                print('target distance:', ptx, pty)
                output_image = cvf.paint_image(output_image, ptx, pty, radius_center_point=10,
                                               radius_delta_1=25, radius_delta_2=45)
                cv2.line(output_image, (int(w / 2), 0), (int(w / 2), h), (0, 255, 255), 3)
                cv2.line(output_image, (0, int(h / 2)), (w, int(h / 2)), (0, 255, 255), 3)
                cv2.circle(output_image, (int(w / 2), int(h / 2)), 90, (0, 0, 255), 3)

                # center of the image
                cv2.rectangle(output_image, (int(h / 2) - 3, int(w / 2) - 3), (int(h / 2) + 3, int(w / 2) + 3),
                              (0, 255, 255), -1)

            cv2.imshow('video', output_image)

        if key == ord('q'):
            stop_z = mc.serial_request(arduino_port_1)[2] / 800
            mc.serial_actuate(0, 0, 0, arduino_port_1)
            break

    cap.release()
    cv2.destroyAllWindows()




if __name__ == "__main__":

    parser = argparse.ArgumentParser('Run Experiments')
    controls_strategies_avialable = ['discrete_jacobian', 'naive', 'potential_field', 'update_jacobian', 'update_jacobian_potential_field']
    parser.add_argument('--control_strategy', required=True,
                        help='control strategy: discrete_jacobian, naive, potential_field, update_jacobian, update_jacobian_potential_field')
    parser.add_argument('--neural_network_dir', required=False,
                        metavar="str", default=os.getcwd(),
                        help='Directory where the tensorflow model to make predictions of images is saved')

    args = parser.parse_args()
    if args.control_strategy in controls_strategies_avialable:

        run_experiment(control_strategy=args.control_strategy)
    else:
        print('control strategy not found, please use one of the following:', controls_strategies_avialable)