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


def get_ground_truth_path(trajectory=3, number_of_times=10):
    type_trajectory = ['straight_line', 'right_curve', 'left_curve', 'right_s', 'left_s']
    SETTINGS = {
        "tracker type": "aurora",
        "romfiles": [''.join([os.getcwd(), '/scripts/em_tracking/080082.rom'])]
    }
    print('connecting with sensor...')

    TRACKER = NDITracker(SETTINGS)
    TRACKER.start_tracking()

    port_arduino = find_arduino.find_arduino()
    arduino_port_1 = mc.serial_initialization(arduino_com_port_1=str(port_arduino))
    print('ATLASCOPE detected at port:', port_arduino)
    times = 0
    flag = False
    while times < number_of_times:

        cap = cv2.VideoCapture(0)
        port_handles, timestamps, framenumbers, tracking, quality = TRACKER.get_frame()
        print(tracking[0][1][3])
        z_now = mc.serial_request(arduino_port_1)[2] / 800

        if tracking[0][1][3] > 5.5 and tracking[0][1][3] != np.nan and flag is False:
            z_now = z_now + 2
            mc.serial_actuate(0, 0, z_now, arduino_port_1)
            print('moving forward')

        if tracking[0][1][3] <= 5.5 and tracking[0][1][3] != np.nan:
            flag = True
            z_now = mc.serial_request(arduino_port_1)[2] / 800
            mc.serial_actuate(0, 0, z_now, arduino_port_1)
            times = times + 1
            print('WAITING')
            for j in range(20):
                print(j)
                print('t minus:', 20-j)
                sleep(0.5)

            sensor_points_x = []
            sensor_points_y = []
            sensor_points_z = []
            time_line = []
            date_experiment = datetime.datetime.now()
            z_vals = []

            while cap.isOpened():
                z = mc.serial_request(arduino_port_1)[2] / 800
                z = z - 2
                z_vals.append(z)
                ret, frame = cap.read()
                cv2.imshow('video', frame)
                key = cv2.waitKey(1) & 0xFF
                port_handles, timestamps, framenumbers, tracking, quality = TRACKER.get_frame()
                mc.serial_actuate(0, 0, z, arduino_port_1)
                act = mc.serial_request(arduino_port_1)
                for i, t in enumerate(tracking):
                    sensor_points_x.append(t[0][3])
                    sensor_points_y.append(t[1][3])
                    sensor_points_z.append(t[2][3])

                    time_line.append(datetime.datetime.now())

                if sensor_points_y[-1] >= 126.5 and sensor_points_z[-1] != np.nan:
                    z_stop = mc.serial_request(arduino_port_1)[2]/800
                    mc.serial_actuate(0, 0, z_stop-1, arduino_port_1)
                    break

                if key == ord('q'):
                    break

            data_vector = [time_line,
                           sensor_points_x,
                           sensor_points_y,
                           sensor_points_z,
                           ]
            dm.save_data_sensors(data_vector, date_experiment, type_trajectory[trajectory])
            print('test:', times, 'ended')
            z_stop = mc.serial_request(arduino_port_1)[2] / 800
            mc.serial_actuate(0, 0, z_stop, arduino_port_1)
            for j in range(20):
                print(j)
                print('t minus:', 20 - j)
                sleep(0.5)
                flag = False
            cap.release()
            cv2.destroyAllWindows()

    TRACKER.stop_tracking()
    TRACKER.close()


def general_calibration():
    from tqdm import tqdm
    port_arduino = find_arduino.find_arduino()
    arduino_port_1 = mc.serial_initialization(arduino_com_port_1=str(port_arduino))
    print('ATLASCOPE detected at port:', port_arduino)
    print('Calibration ongoing:')
    mc.serial_actuate(5, 5, 0, arduino_port_1)
    for i in tqdm(range(1000)):
        sleep(0.01)
    sleep(1)
    mc.serial_actuate(-5, -5, 0, arduino_port_1)
    sleep(1)
    print('Passed!')
    print('Actuators testing:')

    #mc.serial_actuate(0, 10, 0, arduino_port_1)
    #sleep(1)
    #mc.serial_actuate(0, -10, 0, arduino_port_1)
    for i in tqdm(range(1000)):
        sleep(0.02)
    print('Passed!')
    print('EM Sensor calibration:')
    for i in tqdm(range(100)):
        sleep(0.01)
    print('Passed!')
    mc.serial_actuate(0, 0, 10, arduino_port_1)
    sleep(2)
    print('No new version of software found, robot ready to go!')


def determine_q_function():
    directory_data = os.getcwd() + '/data/calibration/'
    cf.plot_data_equivalence(directory_data)


def test_home_coming():

    port_arduino = find_arduino.find_arduino()
    print('Arduino detected at:', port_arduino)
    arduino_port_1 = mc.serial_initialization(arduino_com_port_1=str(port_arduino))

    while True:
        x = float(input("x: "))
        y = float(input("y: "))
        z = float(input("z: "))

        mc.serial_actuate(x, y, z, arduino_port_1)
        print('updated')


def simple_test():

    cap = cv2.VideoCapture(0)
    port_arduino = find_arduino.find_arduino()
    print('Arduino detected at:', port_arduino)
    arduino_port_1 = mc.serial_initialization(arduino_com_port_1=str(port_arduino))
    points_x = list(range(-90, 90))
    points_x = [x/1 for x in points_x]
    previous_velocity = 0
    count = 0
    while cap.isOpened():
        key = cv2.waitKey(1) & 0xFF
        ret, frame = cap.read()

        if ret is True:
            cv2.imshow('video', frame)
            x = points_x[count]
            print(x)
            if x in range(-91, 0):
                velocity = -15
                print('right')
                if previous_velocity != velocity:
                    mc.serial_actuate(velocity, 0, 0, arduino_port_1)

            elif x in range(1, 91):
                velocity = 15
                print('left')
                if previous_velocity != velocity:
                    mc.serial_actuate(velocity, 0, 0, arduino_port_1)

        #sleep(0.02)
        previous_velocity = velocity
        count = count + 1
        if count == len(points_x):
            count = 0
            points_x = points_x[::-1]

        if key == ord('q'):
            mc.serial_actuate(0, 0, 0, arduino_port_1)
            break


def nasty_test():
    SETTINGS = {
        "tracker type": "aurora",
        "romfiles": [''.join([os.getcwd(), '/scripts/em_tracking/080082.rom'])]
    }
    print('connecting with sensor...')
    TRACKER = NDITracker(SETTINGS)
    TRACKER.start_tracking()

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    date_experiment = datetime.datetime.now()

    results_folder = ''.join([os.getcwd(), '/results/experiment_', date_experiment.strftime("%d_%m_%Y_%H_%M"), '/'])
    if not os.path.isdir(results_folder):
        os.mkdir(results_folder)

    name_video = results_folder + 'experiment_lumen_output_video' + date_experiment.strftime("%d_%m_%Y_%H_%M")
    out = cv2.VideoWriter(name_video + '_.avi', fourcc, 20.0, (300, 300))
    cap = cv2.VideoCapture(0)
    port_arduino = find_arduino.find_arduino()
    print('Arduino detected at:', port_arduino)
    arduino_port_1 = mc.serial_initialization(arduino_com_port_1=str(port_arduino))
    initial_z = mc.serial_request(arduino_port_1)[2]/800
    current_act_z = mc.serial_request(arduino_port_1)[2]/800
    old_theta = 0
    old_magnitude = 0
    # directory where the model is located
    project_folder = '/home/nearlab/Jorge/current_work/lumen_segmentation/data/' \
                     'phantom_lumen/results/ResUnet/'
    folder_name = 'ResUnet_lr_0.0001_bs_16_rgb_19_05_2021_17_07'
    # load the model and get input size of the model
    model, input_size = cm.load_model(project_folder, folder_name)
    center_points_x = []
    center_points_y = []
    sensor_points_x = []
    sensor_points_y = []
    sensor_points_z = []
    time_line = []
    joint_variable_values = []
    target_vectors = []
    thetas = []
    magnitudes = []
    actuators_values = []
    filtered_points_x = []
    filtered_points_y = []
    jacobian_matrices = []
    frame_rate = 60
    acumulated_time = datetime.timedelta(seconds=0)
    jacobian_matrix = [[0.33, -0.26], [0.037, 0.81]]
    while cap.isOpened():
        key = cv2.waitKey(1) & 0xFF
        prev = 0
        init_time_epoch = datetime.datetime.now()
        ret, frame = cap.read()
        time_elapsed = time.time() - prev
        port_handles, timestamps, framenumbers, tracking, quality = TRACKER.get_frame()
        if ret is True:
            time_line.append(datetime.datetime.now())
            if time_elapsed > 1. / frame_rate:
                output_image, point_x, point_y = cm.detect_lumen(model, frame)
                #output_image, point_x, point_y = cvf.detect_corners_chessboard(frame)
                center_points_x.append(point_x)
                center_points_y.append(point_y)
                for i, t in enumerate(tracking):
                    sensor_points_x.append(t[0][3])
                    sensor_points_y.append(t[1][3])
                    sensor_points_z.append(t[2][3])

                ptx = dm.calculate_average_points(center_points_x[-6:])
                pty = dm.calculate_average_points(center_points_y[-6:])
                filtered_points_x.append(ptx)
                filtered_points_y.append(pty)
                # if a point is detected
                current_act_joint_variable = mc.serial_request(arduino_port_1)
                joint_variable_values.append(current_act_joint_variable)
                # On-line update Jacobian conrtol

                """if not (np.isnan(ptx)) and not (np.isnan(pty)):
                    print('detected')
                    cv2.circle(output_image, (int(point_x), int(point_y)), 10, (0, 0, 255), -1)
                    h, w, d = np.shape(output_image)
                    if len(joint_variable_values) > 2 and len(filtered_points_x) > 2:
                        jacobian_matrix = gcf.update_jacobian(jacobian_matrix, joint_variable_values[-2:], ptx, pty,
                                                              filtered_points_x[-2], filtered_points_y[-2])

                    target_vector, theta, magnitude = gcf.update_jacobian_control(jacobian_matrix, ptx, pty, (h, w))
                    jacobian_matrices.append(jacobian_matrix)

                    target_vectors.append(target_vector)
                    thetas.append(theta)
                    magnitudes.append(magnitude)

                    delta_time = (datetime.datetime.now()-init_time_epoch)
                    if acumulated_time > datetime.timedelta(seconds=0.1) and magnitude > 0:
                        act = mc.serial_actuate(target_vector[0], target_vector[1], current_act_z, arduino_port_1)
                        acumulated_time = datetime.timedelta(seconds=0)
                        actuators_values.append(act)

                    elif magnitude == 0:
                        current_act_z = (mc.serial_request(arduino_port_1)[2]/800) + 2
                        act = mc.serial_actuate(0, 0, current_act_z, arduino_port_1)
                        acumulated_time = acumulated_time + delta_time
                        actuators_values.append(act)
                    else:
                        acumulated_time = acumulated_time + delta_time
                        actuators_values.append([np.nan, np.nan, np.nan])

                else:
                    actuators_values.append([np.nan, np.nan, np.nan])
                    target_vectors.append([np.nan, np.nan])
                    thetas.append(np.nan)
                    magnitudes.append(np.nan)
                    print('no target detected, stop')
                    mc.serial_actuate(0, 0, current_act_z, arduino_port_1)"""


                # Constant Jacobian control and descreet control just change the control function
                """if not (np.isnan(ptx)) and not (np.isnan(pty)):
                    print('detected')
                    cv2.circle(output_image, (int(point_x), int(point_y)), 10, (0, 0, 255), -1)
                    h, w, d = np.shape(output_image)
                    target_vector, theta, magnitude = gcf.discrete_jacobian_control(ptx, pty, (h, w))
                    target_vectors.append(target_vector)
                    thetas.append(theta)
                    magnitudes.append(magnitude)

                    delta_time = (datetime.datetime.now()-init_time_epoch)
                    if acumulated_time > datetime.timedelta(seconds=0.1) and magnitude > 0:
                        act = mc.serial_actuate(target_vector[0], target_vector[1], current_act_z, arduino_port_1)
                        acumulated_time = datetime.timedelta(seconds=0)
                        actuators_values.append(act)
                    elif magnitude == 0:
                        current_act_z = current_act_z + 1
                        act = mc.serial_actuate(0, 0, current_act_z, arduino_port_1)
                        acumulated_time = acumulated_time + delta_time
                        actuators_values.append(act)
                    else:
                        acumulated_time = acumulated_time + delta_time
                        actuators_values.append([np.nan, np.nan, np.nan])

                else:
                    actuators_values.append([np.nan, np.nan, np.nan])
                    target_vectors.append([np.nan, np.nan])
                    thetas.append(np.nan)
                    magnitudes.append(np.nan)
                    print('no target detected, stop')
                    mc.serial_actuate(0, 0, current_act_z, arduino_port_1)"""

                # Really nasty control
                if not (np.isnan(ptx)) and not (np.isnan(pty)):
                    print('detected')
                    cv2.circle(output_image, (int(point_x), int(point_y)), 10, (0, 0, 255), -1)
                    h, w, d = np.shape(output_image)
                    target_vector, theta, magnitude = gcf.nasty_control(ptx, pty, (h, w))
                    target_vectors.append(target_vector)
                    thetas.append(theta)
                    magnitudes.append(magnitude)
                    if theta != old_theta or magnitude != old_magnitude:
                        print('actuate(x, y)')
                        mc.serial_actuate(0, 0, current_act_z, arduino_port_1)
                        sleep(0.2)
                        act = mc.serial_actuate(target_vector[0], target_vector[1], current_act_z, arduino_port_1)
                    if magnitude == 0:
                        print('actuate(z)')
                        current_act_z = mc.serial_request(arduino_port_1)[2]/800 + 2
                        act = mc.serial_actuate(0, 0, current_act_z, arduino_port_1)

                    actuators_values.append(act)
                    old_theta = theta
                    old_magnitude = magnitude
                else:
                    actuators_values.append([np.nan, np.nan, np.nan])
                    target_vectors.append([np.nan, np.nan])
                    thetas.append(np.nan)
                    magnitudes.append(np.nan)
                    print('no target detected, stop')
                    mc.serial_actuate(0, 0, current_act_z, arduino_port_1)
                #    cv2.imshow('video', frame)
                cv2.imshow('video', output_image)
                out.write(output_image)

            #sleep(0.08)

        if key == ord('q'):
            mc.serial_actuate(0, 0, initial_z, arduino_port_1)
            break

    data_vector = [time_line,
                   center_points_x,
                   center_points_y,
                   filtered_points_x,
                   filtered_points_y,
                   sensor_points_x,
                   sensor_points_y,
                   sensor_points_z,
                   joint_variable_values,
                   target_vectors,
                   thetas,
                   magnitudes,
                   actuators_values,
                   jacobian_matrices]
    dm.save_data(data_vector, date_experiment)
    TRACKER.stop_tracking()
    TRACKER.close()
    out.release()
    cap.release()
    cv2.destroyAllWindows()


def manual_control():
    cap = cv2.VideoCapture(0)
    port_arduino = find_arduino.find_arduino()
    print('Arduino detected at:', port_arduino)
    arduino_port_1 = mc.serial_initialization(arduino_com_port_1=str(port_arduino))
    z = 0
    while cap.isOpened():
        key = cv2.waitKey(1) & 0xFF
        ret, frame = cap.read()
        defined_speed = 5
        if ret is True:
            h, w, d = np.shape(frame)
            cv2.line(frame, (int(w/2), 0), (int(w/2), h), (0, 255, 255), 3)
            cv2.line(frame, (0, int(h/2)), (w, int(h/2)), (0, 255, 255), 3)
            cv2.circle(frame, (int(w/2), int(h/2)), 90, (0, 0, 255), 3)
            cv2.imshow('video', frame)
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
                z_stop = mc.serial_request(arduino_port_1)[2]/800
                z = mc.serial_request(arduino_port_1)[2]/800
                mc.serial_actuate(0, 0, z_stop, arduino_port_1)
            #else:
                #mc.serial_actuate(0, 0, z, arduino_port_1)

        sleep(0.08)

        if key == ord('q'):
            sleep(0.2)
            z_stop = mc.serial_request(arduino_port_1)[2]/800
            print('test finished')
            mc.serial_actuate(0, 0, z_stop, arduino_port_1)
            break

    cap.release()
    cv2.destroyAllWindows()


def test_lumen_detection(project_folder='/home/nearlab/Jorge/current_work/lumen_segmentation/data/' \
                     'phantom_lumen/results/ResUnet/', folder_name='ResUnet_lr_0.0001_bs_16_rgb_19_05_2021_17_07',
                         record_video=False, type_data='rgb'):

    type_data = 'npy'
    cap = cv2.VideoCapture(0)
    model, input_size = cm.load_model(project_folder, folder_name)
    frame_rate = 60
    center_points_x = []
    center_points_y = []

    if record_video is True:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter('_output_video.avi', fourcc, 20.0, (300, 300))

    while cap.isOpened():
        init_time = time.time()
        prev = 0
        ret, frame = cap.read()
        time_elapsed = time.time() - prev

        if ret is True:
            if time_elapsed > 1. / frame_rate:
                output_image, point_x, point_y = cm.detect_lumen(model, frame)
                center_points_x.append(point_x)
                center_points_y.append(point_y)
                average_ptx = dm.calculate_average_points(center_points_x[-6:])
                average_pty = dm.calculate_average_points(center_points_y[-6:])

                # if a point is detected
                if not (np.isnan(average_ptx)) and not (np.isnan(average_pty)):
                    # draw the center point of the image
                    cv2.circle(output_image, (int(average_ptx), int(average_pty)), 3, (255, 255, 255), -1)
                    cv2.circle(output_image, (int(average_ptx), int(average_pty)), 5, (0, 0, 0), 2)
                    # cv2.circle(output_image, (int(h_o / 2), int(w_o / 2)), abs_delta, (0, 255, 255), 3)

                if record_video is True:
                    # write the output frame in the output video
                    out.write(output_image)
                cv2.imshow('lumen', output_image)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        else:
            break

        # Release everything if job is finished
    if record_video is True:
        out.release()
    cap.release()
    cv2.destroyAllWindows()


def test_vision_control(detect_scenario='lumen', record_video=True, abs_delta=15):
    # intialize camera
    cap = cv2.VideoCapture(0)
    if record_video is True:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(detect_scenario + '_output_video.avi', fourcc, 20.0, (300, 300))

    # find arduino
    port_arduino = find_arduino.find_arduino()
    print('Arduino detected at:', port_arduino)
    arduino_port = mc.serial_initialization(arduino_com_port_1=str(port_arduino))

    # initialize the Jacobian matrix, k = 0
    new_jacobian = np.array([[0.625, -0.18], [0.08, 1.0]])

    current_act_x = 0
    current_act_y = 0
    current_act_z = 0

    if detect_scenario == 'lumen':

        # read an image to get its size and determine the initial conditions
        ret, frame = cap.read()
        h, w, d = np.shape(frame)
        previous_x = int(w / 2)
        previous_y = int(h / 2)
        current_act_z = 0
        abs_delta = 55
        frame_rate = 30
        # directory where the model is located
        project_folder = '/home/nearlab/Jorge/current_work/lumen_segmentation/data/' \
                         'phantom_lumen/results/ResUnet/'
        folder_name = 'ResUnet_lr_0.0001_bs_16_rgb_19_05_2021_17_07'
        # load the model and get input size of the model
        model, input_size = cm.load_model(project_folder, folder_name)
        center_points_x = []
        center_points_y = []
        while cap.isOpened():
            init_time = time.time()
            prev = 0
            ret, frame = cap.read()
            time_elapsed = time.time() - prev

            if ret is True:
                if time_elapsed > 1. / frame_rate:
                    output_image, point_x, point_y = cm.detect_lumen(model, frame)
                    center_points_x.append(point_x)
                    center_points_y.append(point_y)
                    average_ptx = dm.calculate_average_points(center_points_x[-6:])
                    average_pty = dm.calculate_average_points(center_points_y[-6:])
                    # if a point is detected
                    if not(np.isnan(average_ptx)) and not(np.isnan(average_pty)):
                        h_o, w_o, d = np.shape(output_image)
                        new_velocity = gcf.less_naive_control(current_act_z, average_ptx,
                                                              average_pty, (h_o, w_o), abs_delta,
                                                              user_define_step=0.08)
                        mc.serial_actuate(new_velocity[0], new_velocity[1], new_velocity[2], arduino_port)
                        # delay so the board has time to receive the signal
                        sleep(0.05)
                        current_act_z = new_velocity[2]
                        # draw the center point of the image
                        cv2.circle(output_image, (int(average_ptx), int(average_pty)), 3, (255, 255, 255), -1)
                        cv2.circle(output_image, (int(average_ptx), int(average_pty)), 5, (0, 0, 0), 2)
                        #cv2.circle(output_image, (int(h_o / 2), int(w_o / 2)), abs_delta, (0, 255, 255), 3)
                        previous_x = average_ptx
                        previous_x = average_pty

                    if record_video is True:
                        # write the output frame in the output video
                        out.write(output_image)
                    cv2.imshow('lumen', output_image)

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        mc.serial_actuate(0, 0, 0, arduino_port)
                        break
            else:
                break

        # Release everything if job is finished
        if record_video is True:
            out.release()
        cap.release()
        cv2.destroyAllWindows()

    elif detect_scenario == 'chessboard':

        # read an image to get its size and determine the initial conditions
        ret, frame = cap.read()
        h, w, d = np.shape(frame)
        previous_x = int(w/2)
        previous_y = int(h/2)
        current_act_z = 0
        abs_delta = 40
        while cap.isOpened():
            # read a frame
            ret, frame = cap.read()
            h, w, d = np.shape(frame)
            if ret is True:
                # detect the corners of the chessboard
                output_image, point_x, point_y = cvf.detect_corners_chessboard(frame)
                # if a point is detected
                if point_x != 'Nan':
                    # send a signal to the board
                    new_velocity = gcf.less_naive_control(current_act_z, point_x,
                                                          point_y, (w, h), abs_delta,
                                                          user_define_step=0.08)
                    mc.serial_actuate(new_velocity[0], new_velocity[1], new_velocity[2], arduino_port)
                    # delay so the board has time to receive the signal
                    sleep(0.1)
                    current_act_z = new_velocity[2]
                    # draw the center point of the chessboard
                    cv2.circle(output_image, (point_x, point_y), 3, (255, 255, 255), -1)
                    cv2.circle(output_image, (point_x, point_y), 5, (0, 0, 0), 2)
                    previous_x = point_x
                    previous_y = point_y
                else:
                    point_x = previous_x
                    point_y = previous_y

                # always draw delta
                cv2.line(output_image, (int(w / 2), int(h / 2)), (point_x, point_y), (255, 0, 0), 3)
                cv2.circle(output_image, (int(w/2), int(h/2)), abs_delta, (0, 255, 255), 3)

                # if the image is grayscale aka 1 channel
                if len(np.shape(output_image)) == 2:
                    # convert to 3 channels image
                    output_image = cv2.cvtColor(output_image, cv2.COLOR_GRAY2BGR)
                if record_video is True:
                    # write the output frame in the output video
                    out.write(output_image)
                # show the output
                cv2.imshow('chessboard', output_image)

            # if the key 'q' is pressed, end loop
            if cv2.waitKey(1) & 0xFF == ord('q'):
                # send a signal to the board to stop the motors
                mc.serial_actuate(0, 0, 0, arduino_port)
                break

        # Release everything if the job is finished
        if record_video is True:
            out.release()
        cap.release()
        cv2.destroyAllWindows()

    elif detect_scenario == 'circle':
        while cap.isOpened():
            ret, frame = cap.read()
            if ret is True:
                h, w, d = np.shape(frame)
                output, points = detect_circle.detect_circle(frame, abs_delta)
                cv2.imshow("output", output)
                # the output from detect circles is in the img ref points with 0,0 in the upper left corner
                print(points[0] - current_act_x)
                print(points[1] - current_act_y)
                new_velocity = gcf.less_naive_control(current_act_z, points[0], points[1],
                                                      (w, h), abs_delta)
                mc.serial_actuate(new_velocity[0], new_velocity[1], new_velocity[2], arduino_port)
                current_act_z = new_velocity[2]

            else:
                cv2.imshow('Circle', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                mc.serial_actuate(0, 0, 0, arduino_port)
                sleep(0.2)
                break


def test_input_arduino():
    port_arduino = find_arduino.find_arduino()
    print('Arduino detected at:', port_arduino)


def test_control():

    predefined_joints_values_cali = [[7, 4, 0],
                                [-6, -2, 0],
                                [3, 2, 0],
                                [-8, -4, 0],
                                [2, 9, 0],
                                [-5, 8, 0],
                                [-10, 2, 0],
                                [-10, 2, 1],
                                [7, 7, 1],
                                [-6, -3, 1],
                                [3, 5, 1],
                                [-6, 8, 1],
                                [3, 2, 1],
                                [-13, 13, 1],
                                [-13, 13, 2],
                                [-4, 4, 2],
                                [8, 3, 2],
                                [8, 13, 2],
                                [-6, -6, 2],
                                [-8, 8, 2],
                                [2, -2, 2],
                                [0, 0, 0]]

    predefined_joints_values = [[0, 0, 0],
                                [0, 0, 2],
                                [0, 0, 0],
                                [0, 0, -2],
                                [0, 0, 0],
                                [20, 0, 0],
                                [-20, 0, 0],
                                [0, 0, 0],
                                [0, 20, 0],
                                [0, -20, 0],
                                [0, 0, 0]]

    port_arduino = find_arduino.find_arduino()
    print('Arduino detected at:', port_arduino)
    arduino_port_1 = mc.serial_initialization(arduino_com_port_1=str(port_arduino))
    c = 100000
    for variable in predefined_joints_values_cali:
        #a = joints_values[0]
        #b = joints_values[1]
        #c = joints_values[2]*5
        a = random.randint(-5, 5)
        b = random.randint(-5, 5)
        c = -c
        print(mc.serial_actuate(0, 0, c, arduino_port_1))
        sleep(0.9)

    print(mc.serial_actuate(0, 0, 0, arduino_port_1))


def test_actuators():
    # TODO: Update to Mega code, use serial_request()
    arduino_port_1, arduino_port_2, arduino_port_3 = pysr.initialize_ports()

    while True:
        motor_enconders = pysr.request_encoder_value(arduino_port_1, arduino_port_2, arduino_port_3)
        encoder_1 = motor_enconders[0]
        encoder_2 = motor_enconders[1]
        encoder_3 = motor_enconders[2]
        print('encoder 1:', encoder_1.decode())
        print('encoder 2:', encoder_2.decode())
        print('encoder 3', encoder_3.decode())


def test_sensor():
    """
    This function returns the points x, y, z from the EM tracking sensor
    """

    SETTINGS = {
        "tracker type": "aurora",
        "romfiles": [''.join([os.getcwd(), '/scripts/em_tracking/080082.rom'])]
    }
    print('connecting with sensor...')
    TRACKER = NDITracker(SETTINGS)
    TRACKER.start_tracking()

    while True:
        emt.test(TRACKER)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    TRACKER.stop_tracking()
    TRACKER.close()


def test_camera():
    """
    This function returns video from the first webcam to the computer.
    The outpus size is fixed to 600 x 300 pixels
    """
    # This will return video from the first webcam on your computer.
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print('Cannot open camera')
        exit()

    while True:
        # reads frames from a camera
        # ret checks return at each frame
        ret, frame = cap.read()
        if not ret:
            print('Cannot recieve frame (stream end?). Exiting')
            break
        w, h, d = np.shape(frame)
        # The original input frame is shown in the window
        cv2.line(frame, (0, int(w/2)), (h, int(w/2)), (0, 255, 255), 3)
        cv2.imshow('Original', frame)
        # Wait for 'a' key to stop the program
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    # Close the window / Release webcam
    cap.release()
    # De-allocate any associated memory usage
    cv2.destroyAllWindows()


def test_all_systems_together():

    # Settings for the em tracking sensor
    SETTINGS={
        "tracker type": "aurora",
        "romfiles": [os.getcwd() + "/scripts/em_tracking/080082.rom"],
    }
    TRACKER = NDITracker(SETTINGS)
    TRACKER.start_tracking()
    # Setting up the video recording settings for the camera
    cap = cv2.VideoCapture(0)
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.avi', fourcc, 20.0, (300, 300))

    while True:
        port_handles, timestamps, framenumbers, tracking, quality = TRACKER.get_frame()
        if not ret:
            print('Cannot recieve frame (stream end?). Exiting')
            break

        for i, t in enumerate(tracking):
            print(i)
            x = t[0][3]
            y = t[1][3]
            z = t[2][3]
            print(x, y, z)

        # reads frames from a camera
        # ret checks return at each frame
        ret, frame = cap.read()
        # read serial to obtain current states of actuation joint variables
        motor_encoders = pysr.request_encoder_value()  # motor_encoders contains is a list contain the current encoder value
        print(motor_encoders)
        # frame is converted to hsv
        resized = cv2.resize(frame, (500, 300))
        # The original input frame is shown in the window
        cv2.imshow('Output Camera', resized)

        # Wait for 'a' key to stop the program
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

        # send serial for new set points for PID controller


    # stop tracking
    TRACKER.stop_tracking()
    TRACKER.close()
    # Close the window / Release webcam
    cap.release()
    # After we release our webcam, we also release the output
    out.release()
    # De-allocate any associated memory usage
    cv2.destroyAllWindows()


def hand_eye_calibration(dir_data):
    if len(os.listdir(dir_data + '/image_list/')) == 0:
       raise ValueError('The directory indicated is empty, '
             'please check the path or use acquire_data_camera_calibration'
             'function to acquire data')

    else:
        cal.hand_eye_calibration(dir_data)


def acquire_data_camera_calibration(save_dir):

    # Settings for the em tracking sensor
    SETTINGS = {
        "tracker type": "aurora",
        "romfiles": [os.getcwd() + "/scripts/em_tracking/080082.rom"],
    }
    TRACKER = NDITracker(SETTINGS)
    TRACKER.start_tracking()

    # This will return video from the first webcam on your computer.
    cap = cv2.VideoCapture(0)
    save_dir = os.getcwd() + '/data/calibration/'
    save_imgs_dir = save_dir + '/image_list/'
    save_pos_dir = save_dir + '/pos_list/'
    print(os.path.isdir(save_dir + '/pos_list/'))
    img_id = 0000

    while True:
        # reads frames from a camera
        # ret checks return at each frame
        ret, frame = cap.read()
        if not ret:
            print('Cannot recieve frame (stream end?). Exiting')
            break

        port_handles, timestamps, framenumbers, tracking, quality = TRACKER.get_frame()

        pattern_size = (7, 5)
        # Converts to grayscale color space
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, corners = cvf.find_corners(gray, pattern_size)
        draw_img = cvf.draw_corners(gray, corners, pattern_size)
        resized = cv2.resize(draw_img, (450, 300))
        # The original input frame is shown in the window
        cv2.imshow('Output Video', resized)

        # Wait for 'a' key to stop the program
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            array_name = ''.join(['calibration_', str(img_id).zfill(4), '.txt'])
            file_name = os.path.join(save_pos_dir, array_name)
            print(file_name)
            for t in tracking:
                #file_array = open(file_name, 'w')
                with open(file_name, "w") as txt_file:
                    txt_file.write(np.array2string(t))
                print(t)

            img_name = ''.join(['calibration_img_', str(img_id).zfill(4), '.png'])
            print(img_name)
            cv2.imwrite(os.path.join(save_imgs_dir, img_name), gray)
            img_id = img_id + 1
        if key == ord('q'):
            break

    # Close the window / Release webcam
    cap.release()
    # De-allocate any associated memory usage
    cv2.destroyAllWindows()
    # stop tracking
    TRACKER.stop_tracking()
    TRACKER.close()
    # Close the window / Release webcam
    cap.release()
    # De-allocate any associated memory usage
    cv2.destroyAllWindows()


if __name__ == "__main__":

    parser = argparse.ArgumentParser('Camera and Hand-Eye robot calibration')
    parser.add_argument('command', metavar="<command>", default='test_camera',
                        help='test_camera, test_sensor, eye_hand_calibration ')
    parser.add_argument('--pattern_size', required=False,
                        metavar="tuple", default=(7, 5),
                        help='Pattern size of the chessboard to detect')
    parser.add_argument('--square_size', required=False,
                        metavar="int", default=0.036,
                        help='Square size of the pattern to detect')
    parser.add_argument('--output_dir', required=False,
                        metavar="int", default=os.getcwd(),
                        help='Directory to save the snapshots and position matrix')
    parser.add_argument('--input_dir', required=False,
                        metavar="int", default=os.getcwd(),
                        help='Directory where the directories /image_list and '
                             '/pos_list are located')

    args = parser.parse_args()
    # Validate arguments
    if args.command == "acquire_data_camera_calibration":
        assert args.pattern_size, "Argument --pattern_size is required for camera " \
                             "calibration and eye-hand calibration"

        assert args.square_size, "Provide --square_size is required for camera " \
                             "calibration and eye-hand calibration"

    #elif args.command == "test_sensor":

    pattern_size = (7, 5)
    square_size = 0.036
    # Configurations
    if args.command == "test_camera":
        test_camera()
    elif args.command == 'test_sensor':
        test_sensor()
    elif args.command == "test_actuators":
        test_actuators()
    elif args.command == "acquire_data_camera_calibration":
        acquire_data_camera_calibration(args.output_dir)
    elif args.command == 'hand_eye_calibration':
        hand_eye_calibration(args.input_dir)
    elif args.command == 'test_control':
        test_control()
    elif args.command == 'test_input_arduino':
        test_input_arduino()
    elif args.command == 'test_vision_control':
        test_vision_control()
    elif args.command == 'test_lumen_detection':
        test_lumen_detection()
    elif args.command == 'home_coming':
        test_home_coming()
    elif args.command == 'determine_q_function':
        determine_q_function()
    elif args.command == 'general_calibration':
        general_calibration()
    elif args.command == 'manual_control':
        manual_control()
    elif args.command == 'simple_test':
        simple_test()
    elif args.command == 'nasty_test':
        nasty_test()
    elif args.command == 'get_ground_truth_path':
        get_ground_truth_path()

    else:
        raise Exception("The command written was not found")


"""    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        train(model, args.dataset, args.subset)
    elif args.command == "detect":
        detect(model, args.dataset, args.subset)
    elif args.command == "splash":
        detect_and_color_splash(model, image_path=args.image)
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'detect'".format(args.command))

    calibrate_camera()"""
