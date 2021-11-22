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


def run_experiment(control_strategy):

    # Define the variables necessary to initialize the EM tracing sensor
    SETTINGS = {
        "tracker type": "aurora",
        "romfiles": [''.join([os.getcwd(), '/scripts/em_tracking/080082.rom'])]
    }
    print('connecting with sensor...')
    TRACKER = NDITracker(SETTINGS)
    TRACKER.start_tracking()

    # Initialize the video write
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    date_experiment = datetime.datetime.now()

    # Define the results saving directory
    results_folder = ''.join([os.getcwd(), '/results/experiment_', control_strategy, '_', date_experiment.strftime("%d_%m_%Y_%H_%M"), '/'])
    # If the folder doesn't exists create the folder
    if not os.path.isdir(results_folder):
        os.mkdir(results_folder)

    # Define the name od the video to save
    name_video = results_folder + 'experiment_lumen_output_video' + control_strategy + '_' + date_experiment.strftime("%d_%m_%Y_%H_%M")
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

    # directory where the model for lumen segmentation is located
    project_folder = os.getcwd() + '/scripts/computer_vision/models/weights/'
    folder_model = 'Transpose_ResUnet_lr_0.001_bs_16_rgb_27_05_2021_13_03'
    # load the model and get input size of the model
    model, input_size = cm.load_model(project_folder, folder_model)
    print('input size', input_size)
    # Define all the empty arrays of the variables that will be saved
    # Points detected by the CV module
    center_points_x = []
    center_points_y = []
    # Sensor data
    sensor_1_points_x = []
    sensor_1_points_y = []
    sensor_1_points_z = []
    sensor_2_points_x = []
    sensor_2_points_y = []
    sensor_2_points_z = []
    sensor_3_points_x = []
    sensor_3_points_y = []
    sensor_3_points_z = []
    # limit points defined in
    limit_points_x = []
    limit_points_y = []
    limit_points_z = []
    # time
    time_line = []
    # values from the motors encoders
    joint_variable_values = []
    # target vectors end to the encoders
    target_vectors = []
    # angle values
    thetas = []
    # magnitudes of the vectors
    magnitudes = []
    actuators_values = []

    filtered_points_x = []
    filtered_points_y = []

    jacobian_matrices = []
    acumulated_time = datetime.timedelta(seconds=0)
    delta_time = datetime.timedelta(seconds=0)
    jacobian_matrix = [[0.83, -0.02], [0.038, 1.01]]
    counter = 0
    print('Please input 6 points for each corner of the trajectory to follow')

    # Run the experiment
    while cap.isOpened():
        # read a key from the keyboard
        key = cv2.waitKey(1) & 0xFF
        # get the frame from the camera
        ret, frame = cap.read()
        counter += 1
        # get the data from the EM tracking system
        port_handles, timestamps, framenumbers, tracking, quality = TRACKER.get_frame()

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

            # save sensor points
            sensor_1_points_x.append(tracking[0][0][3])
            sensor_1_points_y.append(tracking[0][1][3])
            sensor_1_points_z.append(tracking[0][2][3])

            sensor_2_points_x.append(tracking[1][0][3])
            sensor_2_points_y.append(tracking[1][1][3])
            sensor_2_points_z.append(tracking[1][2][3])

            sensor_3_points_x.append(tracking[2][0][3])
            sensor_3_points_y.append(tracking[2][1][3])
            sensor_3_points_z.append(tracking[2][2][3])

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
            jacobian_matrices.append(jacobian_matrix)
            # increase the accumulated time
            acumulated_time = acumulated_time + delta_time

            # in case a point was detected by the CV module
            if not (np.isnan(ptx)) and not (np.isnan(pty)):
                h, w, d = np.shape(output_image)
                # Draw circunference around the points
                output_image = cvf.paint_image(output_image, ptx, pty, radius_center_point=10,
                                               radius_delta_1=25, radius_delta_2=45)
                # draw a rectangle in the center of the image
                cv2.rectangle(output_image, (int(h / 2) - 3, int(w / 2) - 3), (int(h / 2) + 3, int(w / 2) + 3),
                              (0, 255, 255), -1)

                # calculate the time passed since last action
                delta_time = (datetime.datetime.now()-init_time_epoch)
                # if bigger than some threshold calculate the actuation variables
                # according to the control algorithm
                if acumulated_time > datetime.timedelta(seconds=0.1):
                    if control_strategy == 'discrete_jacobian':
                        target_vector, theta, magnitude = gcf.discrete_jacobian_control(ptx, pty, (h, w), jacobian_matrix)
                    elif control_strategy == 'potential_field':
                        print('time:', acumulated_time.total_seconds())
                        target_vector, theta, magnitude = gcf.potential_field(ptx, pty, (h, w),
                                                                              acumulated_time.total_seconds(),
                                                                              delta_border=30,
                                                                              jacobian=jacobian_matrix)
                        cv2.circle(output_image, (int(h / 2), int(w / 2)), 30,
                                   (0, 255, 0), 2)
                    elif control_strategy == 'naive':
                        target_vector, theta, magnitude = gcf.nasty_control(ptx, pty, (h, w))
                    elif control_strategy == 'update_jacobian':
                        jacobian_matrix = gcf.update_jacobian(jacobian_matrix, joint_variable_values[-2:],
                                                          ptx, pty, filtered_points_x[-1], filtered_points_y[-1])
                        target_vector, theta, magnitude = gcf.discrete_jacobian_control(ptx, pty, (h, w), jacobian_matrix)
                    elif control_strategy == 'update_jacobian_potential_field':
                        jacobian_matrix = gcf.update_jacobian(jacobian_matrix, joint_variable_values[-2:],
                                                          ptx, pty, filtered_points_x[-1], filtered_points_y[-1])
                        target_vector, theta, magnitude = gcf.potential_field(ptx, pty, (h, w),
                                                                              acumulated_time.total_seconds(),
                                                                              delta_border=30,
                                                                              jacobian=jacobian_matrix)
                    # save the target vectors, the angle and the magnitude
                    target_vectors.append(target_vector)
                    thetas.append(theta)
                    magnitudes.append(magnitude)
                    # in case the magnitude of the error is bigger than some threshold,
                    # actuate the motors for position in x and y otherwise actuate displacement
                    if magnitude > 0:
                        print('actuate(x, y)')
                        # the next 3 lines seems to avoid the robot to acumulate instruction and de-stabilize
                        current_act_z = mc.serial_request(arduino_port_1)[2] / 800
                        mc.serial_actuate(0, 0, current_act_z, arduino_port_1)
                        sleep(0.1)
                        act = mc.serial_actuate(target_vector[0], target_vector[1], current_act_z, arduino_port_1)
                        actuators_values.append(act)
                    elif magnitude == 0:
                        print('actuate(z)')
                        current_act_z = mc.serial_request(arduino_port_1)[2]/800 + 2
                        act = mc.serial_actuate(target_vector[0], target_vector[1], current_act_z, arduino_port_1)
                        #act = mc.serial_actuate(0, 0, current_act_z, arduino_port_1)
                        actuators_values.append(act)

                    else:
                        print('true')
                        actuators_values.append([np.nan, np.nan, np.nan])

                    acumulated_time = datetime.timedelta(seconds=0)
                # In case there is no actuation keep ans record previous values
                else:
                    if len(target_vectors) > 1:
                        actuators_values.append(actuators_values[-1])
                        target_vectors.append(target_vectors[-1])
                        thetas.append(thetas[-1])
                        magnitudes.append(magnitudes[-1])
                    else:
                        actuators_values.append([np.nan, np.nan, np.nan])
                        target_vectors.append([np.nan, np.nan])
                        thetas.append(np.nan)
                        magnitudes.append(np.nan)

            # In case no point detected keep and record previous values
            else:
                actuators_values.append([np.nan, np.nan, np.nan])
                target_vectors.append([np.nan, np.nan])
                thetas.append(np.nan)
                magnitudes.append(np.nan)
                print('no target detected, stop')
                mc.serial_actuate(0, 0, current_act_z, arduino_port_1)
            # show video
            cv2.imshow('video', output_image)
            # save video
            out.write(output_image)

        else:
            print("please collect the calibration data, press 's' to save a point")
            cv2.imshow('video', frame)

        # acquire data for calibration
        if key == ord('s'):
            limit_points_x.append(tracking[3][0][3])
            limit_points_y.append(tracking[3][1][3])
            limit_points_z.append(tracking[3][2][3])
            print('point saved')

        if key == ord('q'):
            stop_z = mc.serial_request(arduino_port_1)[2]/800
            mc.serial_actuate(0, 0, 0, arduino_port_1)
            break

    # arrange all the saved data in an array
    data_vector = [time_line,
                   center_points_x,
                   center_points_y,
                   filtered_points_x,
                   filtered_points_y,
                   sensor_1_points_x,
                   sensor_1_points_y,
                   sensor_1_points_z,
                   sensor_2_points_x,
                   sensor_2_points_y,
                   sensor_2_points_z,
                   sensor_3_points_x,
                   sensor_3_points_y,
                   sensor_3_points_z,
                   joint_variable_values,
                   target_vectors,
                   thetas,
                   magnitudes,
                   actuators_values,
                   jacobian_matrices]
    # stop the robot
    mc.serial_actuate(0, 0, initial_z, arduino_port_1)
    dictionary = {"limit_point_x": limit_points_x,
                  "limit_point_y": limit_points_y,
                  "limit_point_z": limit_points_z}

    # save the calibration data in a Json
    dm.save_data_json(dictionary, results_folder)
    # save the data measurements in a CSV
    dm.save_data(data_vector, results_folder, date_experiment)
    # strop tracking and close
    TRACKER.stop_tracking()
    TRACKER.close()
    # release cam and stop recording video
    out.release()
    cap.release()
    # close all windows
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