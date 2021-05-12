import numpy as np
import cv2
from general_functions import calibration_functions as cf
import argparse
import os


def test_camera():
    """
    This function returns video from the first webcam to the computer.
    The outpus size is fixed to 600 x 300 pixels
    """
    # This will return video from the first webcam on your computer.
    cap = cv2.VideoCapture(0)
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.avi', fourcc, 20.0, (600, 300))

    while (True):
        # reads frames from a camera
        # ret checks return at each frame
        ret, frame = cap.read()
        # The original input frame is shown in the window
        cv2.imshow('Original', frame)
        # Wait for 'a' key to stop the program
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    # Close the window / Release webcam
    cap.release()
    # After we release our webcam, we also release the output
    out.release()
    # De-allocate any associated memory usage
    cv2.destroyAllWindows()


def calibrate_camera(save_dir, pattern_size, square_size):
    pattern_size = (7, 5)
    square_size = 0.036

    # This will return video from the first webcam on your computer.
    cap = cv2.VideoCapture(0)

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.avi', fourcc, 20.0, (300, 300))
    save_dir = '/home/nearlab/Jorge/current_work/robot_vision/data/calibration/'

    img_id = 0000
    while (True):
        # reads frames from a camera
        # ret checks return at each frame
        ret, frame = cap.read()
        pattern_size = (7, 5)
        square_size = 0.036
        # Converts to HSV color space, OCV reads colors as BGR
        # frame is converted to hsv
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, corners = cf.find_corners(gray, pattern_size)
        draw_img = cf.draw_corners(gray, corners, pattern_size)

        resized = cv2.resize(draw_img, (450, 300))
        # The original input frame is shown in the window
        cv2.imshow('Original', resized)

        # Wait for 'a' key to stop the program
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            img_name = 'calibration_img_' + str(img_id).zfill(4) + '.png'
            print(img_name)
            cv2.imwrite(save_dir + img_name, gray)
            img_id = img_id + 1

        if key == ord('q'):
            break

    # Close the window / Release webcam
    cap.release()

    # After we release our webcam, we also release the output
    out.release()

    # De-allocate any associated memory usage
    cv2.destroyAllWindows()


if __name__ == "__main__":

    parser = argparse.ArgumentParser('Camera and Hand-Eye robot calibration')
    parser.add_argument('command', metavar="<command>", default='test_camera',
                        help='test_camera, calibrate_camera, eye_hand_calibration ')
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
    if args.command != "test_camera":
        assert args.pattern_size, "Argument --pattern_size is required for camera " \
                             "calibration and eye-hand calibration"
    elif args.command == "detect":
        assert args.square_size, "Provide --square_size is required for camera " \
                             "calibration and eye-hand calibration"

    pattern_size = (7, 5)
    square_size = 0.036
    # Configurations
    if args.command == "test_camera":
        test_camera()
    else:
        calibrate_camera(args.save_dir, args.pattern_size, args.square_size)

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