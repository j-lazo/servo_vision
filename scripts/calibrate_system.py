import numpy as np
import cv2
from general_functions import calibration_functions as cf
import argparse
import os
from computer_vision import calibration
from em_tracking import sample_test as emt
from sksurgerynditracker.nditracker import NDITracker
from control import pySerial_request as pysr


def test_actuators():

    arduino_port_1, arduino_port_2, arduino_port_3 = pysr.initialize_ports()

    while True:
        motor_enconders = pysr.request_encoder_value(arduino_port_1, arduino_port_2, arduino_port_3)
        encoder_1 = motor_enconders[0]
        encoder_2 = motor_enconders[1]
        encoder_3 = motor_enconders[2]
        print('\r' in ((encoder_1.decode())))


def test_sensor():
    """
    This function returns the points x, y, z from the EM tracking sensor
    """

    SETTINGS = {
        "tracker type": "aurora",
        "romfiles": [''.join([os.getcwd(), '/scripts/em_tracking/080082.rom'])]
    }

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

        # The original input frame is shown in the window
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
    parser.add_argument('--save_dir', required=False,
                        metavar="int", default=os.getcwd(),
                        help='Directory to save the snapshots')

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
        acquire_data_camera_calibration(args.save_dir)
    elif args.command == 'hand_eye_calibration':
        acquire_data_camera_calibration(args.save_dir)


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
