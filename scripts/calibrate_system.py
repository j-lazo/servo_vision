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
import datetime
import time


def determine_q_function():
    directory_data = '/home/nearlab/Jorge/current_work/robot_vision/data/calibration/'

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


def test_lumen_detection(project_folder='/home/nearlab/Jorge/current_work/lumen_segmentation/data/' \
                     'phantom_lumen/', folder_name='ResUnet_lr_0.0001_bs_16_rgb_19_05_2021_17_07', type_data='rgb'):

    type_data = 'npy'

    cap = cv2.VideoCapture(0)
    new_results_id = folder_name
    results_directory = ''.join([project_folder, 'results/ResUnet/',
                                 new_results_id, '/'])
    name_model = ''.join([results_directory, new_results_id, '_model.h5'])
    print('NAME MODEL')
    print(name_model)
    model = tf.keras.models.load_model(name_model,
                                       custom_objects={'loss': cvf.dice_coef_loss},
                                       compile=False)
    frame_rate = 60
    point_x, point_y = 0, 0

    counter = 0
    while cap.isOpened():
        prev = 0
        counter = counter + 1
        ret, frame = cap.read()
        init_time = time.time()
        time_elapsed = time.time() - prev

        if ret is True and counter > 10:

            if time_elapsed > 1. / frame_rate:
                reshaped = cv2.resize(frame, (256, 256), interpolation=cv2.INTER_AREA)
                resized = cv2.blur(reshaped, (7, 7))/255
                mask = cvf.predict_mask(model, resized)
                resized_2 = cv2.resize(reshaped, (300, 300), interpolation=cv2.INTER_AREA)
                w, h, d = np.shape(resized_2)
                previous_point_x = point_x
                previous_point_y = point_y
                point_x, point_y = cvf.detect_dark_region(mask, resized_2)
                if point_x != 'nAN':
                    cv2.circle(resized_2, (int(point_x), int(point_y)), 45, (0, 0, 255), 2)
                if point_y != 'nAN':
                    cv2.circle(resized_2, (int(point_x), int(point_y)), 25, (0, 0, 255), 2)

                if point_x == 'nAN':
                   point_x = previous_point_x
                if point_y == 'nAN':
                   point_y = previous_point_y

                cv2.line(resized_2, (int(point_x), int(point_y)), (int(w/2), int(h/2)), (255, 0, 0), 4)
                cv2.circle(resized_2, (int(w / 2), int(h / 2)), 3, (0, 0, 255), -1)
                cv2.imshow('frame', resized_2)
                print(point_x, point_y, 1/(time.time()-init_time))
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        else:
            break

    # Release everything if job is finished
    cap.release()
    cv2.destroyAllWindows()


def test_vision_control(detect_scenario='chessboard', abs_delta=70):

    cap = cv2.VideoCapture(0)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.avi', fourcc, 20.0, (300, 300))
    port_arduino = find_arduino.find_arduino()
    print('Arduino detected at:', port_arduino)
    arduino_port = mc.serial_initialization(arduino_com_port_1=str(port_arduino))

    current_act_x = 0
    current_act_y = 0
    current_act_z = 0

    if detect_scenario == 'lumen':

        project_folder = '/home/nearlab/Jorge/current_work/lumen_segmentation/data/' \
                         'phantom_lumen/'

        folder_name = 'ResUnet_lr_0.0001_bs_16_rgb_19_05_2021_17_07'
        new_results_id = folder_name
        results_directory = ''.join([project_folder, 'results/ResUnet/',
                                     new_results_id, '/'])
        name_model = ''.join([results_directory, new_results_id, '_model.h5'])
        print('NAME MODEL')
        print(name_model)
        model = tf.keras.models.load_model(name_model,
                                           custom_objects={'loss': cvf.dice_coef_loss},
                                           compile=False)
        frame_rate = 60
        point_x, point_y = 0, 0

        #initialize the Jacobian matrix, k = 0
        new_jacobian = np.array([[0.625, -0.18], [0.08, 1.0]])
        old_position = [0, 0, 0]
        flag_loop = 0
        while cap.isOpened():
            init_time = time.time()
            prev = 0
            ret, frame = cap.read()
            time_elapsed = time.time() - prev

            if ret is True:

                if time_elapsed > 1. / frame_rate:
                    reshaped = cv2.resize(frame, (256, 256), interpolation=cv2.INTER_AREA)
                    resized = cv2.blur(reshaped, (7, 7)) / 255
                    mask = cvf.predict_mask(model, resized)
                    resized_2 = cv2.resize(reshaped, (300, 300), interpolation=cv2.INTER_AREA)
                    w, h, d = np.shape(resized_2)
                    previous_point_x = point_x
                    previous_point_y = point_y
                    point_x, point_y = cvf.detect_dark_region(mask, resized_2)
                    if point_x != 'nAN':
                        cv2.circle(resized_2, (int(point_x), int(point_y)), 45, (0, 0, 255), 2)
                    if point_y != 'nAN':
                        cv2.circle(resized_2, (int(point_x), int(point_y)), 25, (0, 0, 255), 2)

                    if point_x == 'nAN':
                        point_x = previous_point_x
                    if point_y == 'nAN':
                        point_y = previous_point_y

                    #new_position = gcf.naive_control(current_act_x, current_act_y, current_act_z,
                    #                                 point_x, point_y, (w, h), abs_delta)

                    #mc.serial_actuate(new_position[0], new_position[1], new_position[2], arduino_port)
                    #current_act_x = new_position[0]
                    #current_act_y = new_position[1]
                    #current_act_z = new_position[2]
                    #######original less naive control block############
                    # new_velocity = gcf.less_naive_control(current_act_z, point_x, point_y,
                    #                                       (w, h), abs_delta)
                    # mc.serial_actuate(new_velocity[0], new_velocity[1], new_velocity[2], arduino_port)
                    # current_act_z = new_velocity[2]
                    #######\end original less naive control block############
                    #######jacobian_correction_control
                    if flag_loop == 0:
                        delta_time = float(time.time() - init_time)
                        current_act_joint_variable = mc.serial_request(arduino_port)
                        new_position = gcf.jacobian_correction_velocity_control(new_jacobian, point_x, point_y, (w, h), abs_delta)
                        flag_loop = 1
                    else:

                        delta_q = [0, 0]
                        delta_time = float(time.time() - init_time)
                        current_act_joint_variable = mc.serial_request(arduino_port)
                        delta_q[0] = current_act_joint_variable[0] - old_act_joint_variable[0]
                        delta_q[1] = current_act_joint_variable[1] - old_act_joint_variable[1]
                        new_jacobian = gcf.update_jacobian(new_jacobian, delta_time, delta_q, point_x, point_y, previous_point_x, previous_point_y)
                        new_position = gcf.jacobian_correction_velocity_control(new_jacobian, point_x, point_y, (w, h), abs_delta)
                        print(new_position)
                    mc.serial_actuate(new_position[0], new_position[1], new_position[2], arduino_port)
                    old_position = new_position
                    old_act_joint_variable = current_act_joint_variable

                    #sleep(0.01)
                    cv2.line(resized_2, (int(point_x), int(point_y)), (int(w / 2), int(h / 2)), (255, 0, 0), 4)
                    cv2.circle(resized_2, (int(w / 2), int(h / 2)), 3, (0, 255, 255), -1)
                    #cv2.circle(resized_2, (int(point_x), int(point_y)), 20, (0, 0, 255), 2)
                    # yellow circle
                    #cv2.circle(resized_2, (int(w / 2), int(h / 2)), abs_delta, (0, 255, 255), 2)
                    cv2.imshow('frame', resized_2)
                    out.write(resized_2)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        sleep(0.5)
                        mc.serial_actuate(0, 0, 0, arduino_port)
                        break
            else:
                break

        # Release everything if job is finished
        #mc.serial_actuate(0, 0, 0, arduino_port)
        cap.release()
        out.release()
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
                # new_position = gcf.naive_control(current_act_x, current_act_y, current_act_z,
                #                                  points[0], points[1], (w, h), abs_delta)
                #
                # mc.serial_actuate(new_position[0], new_position[1], new_position[2], arduino_port)
                # current_act_x = new_position[0]
                # current_act_y = new_position[1]
                # current_act_z = new_position[2]
                # sleep(0.1)

                new_velocity = gcf.less_naive_control(current_act_z, points[0], points[1],
                                                      (w, h), abs_delta)
                mc.serial_actuate(new_velocity[0], new_velocity[1], new_velocity[2], arduino_port)
                current_act_z = new_velocity[2]

            else:
                cv2.imshow('test', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    elif detect_scenario == 'chessboard':

        point_x, point_y = 0, 0
        #initialize the Jacobian matrix, k = 0
        new_jacobian = np.array([[0.625, -0.18], [0.08, 1.0]])
        old_position = [0, 0, 0]
        flag_loop = 0
        while cap.isOpened():
            init_time = time.time()
            ret, frame = cap.read()
            if ret is True:
                previous_point_x = point_x
                previous_point_y = point_y
                # Converts to grayscale color space
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                w, h = np.shape(gray)
                #print('target', point_x, point_y)
                if flag_loop == 0:
                    delta_time = float(time.time() - init_time)
                    current_act_joint_variable = mc.serial_request(arduino_port)
                    new_position = gcf.jacobian_correction_velocity_control(new_jacobian, point_x, point_y, (w, h),
                                                                            abs_delta)
                    flag_loop = 1
                else:

                    delta_q = [0, 0]
                    delta_time = float(time.time() - init_time)
                    current_act_joint_variable = mc.serial_request(arduino_port)
                    delta_q[0] = current_act_joint_variable[0] - old_act_joint_variable[0]
                    delta_q[1] = current_act_joint_variable[1] - old_act_joint_variable[1]
                    #new_jacobian = gcf.update_jacobian(new_jacobian, delta_time, delta_q, point_x, point_y,
                    #                                   previous_point_x, previous_point_y)
                    new_position = gcf.jacobian_correction_velocity_control(new_jacobian, point_x, point_y, (w, h),
                                                                            abs_delta)
                    print(new_position)
                    mc.serial_actuate(new_position[0], new_position[1], new_position[2], arduino_port)
                old_position = new_position
                old_act_joint_variable = current_act_joint_variable
                pattern_size = (7, 5)
                found, corners = cf.find_corners(gray, pattern_size)
                draw_img = cf.draw_corners(gray, corners, pattern_size)
                resized = cv2.resize(draw_img, (450, 300))
                # The original input frame is shown in the window
                cv2.imshow('Output Video', resized)

                points_x = []
                points_y = []
                if found:
                    for x in range(np.shape(corners)[0]):
                        points_x.append(corners[x][0][0])
                        points_y.append(corners[x][0][1])

                    if points_x:
                        point_x = points_x[5]
                    if points_y:
                        point_y = points_y[5]

            else:
                cv2.imshow('test', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                mc.serial_actuate(0, 0, 0)
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
    #save_dir = '/home/nearlab/Jorge/current_work/robot_vision/data/calibration/'
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
        found, corners = cf.find_corners(gray, pattern_size)
        draw_img = cf.draw_corners(gray, corners, pattern_size)
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
    elif args.command == 'test_sensors':
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
