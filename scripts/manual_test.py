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
import math


def calculate_accumulated_error(target_x, target_y, img_shape, array_errors):

    transformed_x = -(target_x - (img_shape[0] / 2))
    transformed_y = -(target_y - (img_shape[1] / 2))
    current_error = math.sqrt(transformed_x ** 2 + transformed_y ** 2)
    array_errors.append(current_error)
    accumulated_error = np.mean(array_errors[-20:])

    return accumulated_error, array_errors


def run_experiment(type_experiment, user_id):
    # Initialize the video writer
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # get the current date and time
    date_experiment = datetime.datetime.now()

    # Define the results saving directory
    results_folder = ''.join([os.getcwd(), '/results/experiment_', type_experiment, '_',
                              'usr_', user_id, '_', date_experiment.strftime("%d_%m_%Y_%H_%M"), '/'])

    # If the folder doesn't exists create the folder
    if not os.path.isdir(results_folder):
        os.mkdir(results_folder)

    # Define the name of the video to save
    name_video = results_folder + 'experiment_lumen_output_video' + type_experiment + '_' + \
                 'usr_' + user_id + '_' + date_experiment.strftime("%d_%m_%Y_%H_%M")

    # Define the format of the video and size of the frames
    out = cv2.VideoWriter(name_video + '_.avi', fourcc, 20.0, (300, 300))
    cap = cv2.VideoCapture(0)
    # Initialize Arduino
    port_arduino = find_arduino.find_arduino()
    print('Arduino detected at:', port_arduino)
    arduino_port_1 = mc.serial_initialization(arduino_com_port_1=str(port_arduino))

    # Obtaining the initial value of the encoder of the linear stage
    initial_z = mc.serial_request(arduino_port_1)[2]/800
    current_act_z = mc.serial_request(arduino_port_1)[2]/800

    # Define the limits of the speed of the actuator and the default speed
    max_speed = 35
    mins_speed = 1
    prev_key = 0
    defined_speed = 5
    z = 0
    acumulated_error = 100
    stop_threshold = 1

    # directory where the model for lumen segmentation is located
    project_folder = os.getcwd() + '/scripts/computer_vision/models/weights/'
    folder_model = 'Transpose_ResUnet_lr_0.001_bs_16_rgb_27_05_2021_13_03'
    # load the model and get input size of the model
    model, input_size = cm.load_model(project_folder, folder_model)
    print('input size', input_size)
    # Define all the empty arrays of the variables that will be saved
    # Define all the empty arrays of the variables that will be saved
    # Points detected by the CV module
    center_points_x = []
    center_points_y = []

    # limit points defined in
    limit_points_x = []
    limit_points_y = []
    limit_points_z = []
    # time
    time_line = []
    # values from the motors encoders
    joint_variable_values = []

    filtered_points_x = []
    filtered_points_y = []

    error_values = []
    counter = 0
    print('Please input 6 points for each corner of the trajectory to follow')
    # Run the experiment
    while cap.isOpened():
        # read a key from the keyboard
        key = cv2.waitKey(1) & 0xFF
        # get the frame from the camera
        ret, frame = cap.read()
        counter += 1

        if ret is True and counter > 10 and len(limit_points_x) >= 48:
            # initialize the time counter of the cycle
            init_time_epoch = datetime.datetime.now()
            time_line.append(datetime.datetime.now())
            #if input_size != 3:
            #else:

            # detected points by the nn
            output_image, point_x, point_y = cm.detect_lumen(model, frame)
            # save detected points
            center_points_x.append(point_x)
            center_points_y.append(point_y)

            # Calculate the average of the last 4 points
            ptx = dm.calculate_average_points(center_points_x[-4:])
            pty = dm.calculate_average_points(center_points_y[-4:])
            # save the filtered points
            filtered_points_x.append(ptx)
            filtered_points_y.append(pty)
            # request the current values of the actuators
            current_act_joint_variable = mc.serial_request(arduino_port_1)
            # save the values of the actuators and the Jacobian Matrix
            joint_variable_values.append(current_act_joint_variable)

            # in case a point was detected by the CV module
            if not (np.isnan(ptx)) and not (np.isnan(pty)):
                h, w, d = np.shape(output_image)
                if type_experiment == 'visual_feedback':

                    # Draw circunference around the points
                    output_image = cvf.paint_image(output_image, ptx, pty, radius_center_point=10,
                                                   radius_delta_1=25, radius_delta_2=45)
                    # draw the circles
                    cv2.circle(output_image, (int(h / 2), int(w / 2)), 30,
                               (0, 255, 0), 2)
                    # draw a rectangle in the center of the image
                    cv2.rectangle(output_image, (int(h / 2) - 3, int(w / 2) - 3), (int(h / 2) + 3, int(w / 2) + 3),
                              (0, 255, 255), -1)

                stop_threshold, error_values = calculate_accumulated_error(ptx, pty, (h, w), error_values)
            # In case no point detected keep and record previous values
            else:
                print('no target detected, stop')
                mc.serial_actuate(0, 0, current_act_z, arduino_port_1)
            # show video
            cv2.imshow('video', output_image)
            # save video
            out.write(output_image)


        # keys for the user
        # (+) increase speed
        if key == 43 and defined_speed <= max_speed:
            defined_speed = defined_speed + 1
            print('speed', defined_speed)
            key = prev_key
        # (-) reduce speed
        if key == 45 and defined_speed >= mins_speed:
            defined_speed = defined_speed - 1
            print('speed', defined_speed)

        # (right arrow) move right
        if key == 83:
            print('right')
            mc.serial_actuate(-defined_speed, 0, z, arduino_port_1)
        # (left arrow) move left
        elif key == 81:
            print('left')
            mc.serial_actuate(defined_speed, 0, z, arduino_port_1)
        # (up arrow) move up
        elif key == 82:
            print('up')
            mc.serial_actuate(0, defined_speed, z, arduino_port_1)
        elif key == 84:
            print('down')
        # (down arrow) move down
            mc.serial_actuate(0, -defined_speed, z, arduino_port_1)

        #elif key == ord('f'):
        #    print('forward')
        #    z = z + 1
        #    mc.serial_actuate(0, 0, z, arduino_port_1)
        #elif key == ord('b'):
        #    print('backwards')
        #    z = z - 1
        #    mc.serial_actuate(0, 0, z, arduino_port_1)
        # (s) stop movement
        elif key == ord('s'):
            print('stop')
            z_stop = mc.serial_request(arduino_port_1)[2] / 800
            z = mc.serial_request(arduino_port_1)[2] / 800
            mc.serial_actuate(0, 0, z_stop, arduino_port_1)
        # else:
        # mc.serial_actuate(0, 0, z, arduino_port_1)
        prev_key = key
        # (q) quit program. Finish experiment
        if key == ord('q') or acumulated_error < stop_threshold:
            stop_z = mc.serial_request(arduino_port_1)[2] / 800
            mc.serial_actuate(0, 0, 0, arduino_port_1)
            break

    # arrange all the saved data in an array
    data_vector = [time_line,
                   center_points_x,
                   center_points_y,
                   filtered_points_x,
                   filtered_points_y,
                   joint_variable_values,
                   ]


    # stop the robot
    mc.serial_actuate(0, 0, initial_z, arduino_port_1)
    # save the data measurements in a CSV
    dm.save_data_manual_test(data_vector, results_folder, date_experiment)

    # release cam and stop recording video
    out.release()
    cap.release()
    # close all windows
    cv2.destroyAllWindows()


if __name__ == "__main__":

    parser = argparse.ArgumentParser('Run Experiments')
    type_experiment = ['visual_feedback', 'completely_manual']
    parser.add_argument('--type_experiment', required=True,
                        help='visual_feedback, completely_manual')
    parser.add_argument('--user_id', required=True,
                        metavar="str", default=os.getcwd(),
                        help='Directory where the tensorflow model to make predictions of images is saved')

    args = parser.parse_args()
    if args.control_strategy in type_experiment:

        run_experiment(type_experiment=args.type_experiment, user_id=args.user_id)
    else:
        print('control strategy not found, please use one of the following:', type_experiment)