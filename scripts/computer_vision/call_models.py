import time
import numpy as np
import cv2
from tensorflow.keras.layers import *
import tensorflow as tf
from computer_vision import general_functions as cvf


def load_model(project_folder, name_model):
    results_directory = ''.join([project_folder, name_model, '/'])
    name_model = ''.join([results_directory, name_model, '_model.h5'])
    print('MODEL USED:')
    print(name_model)
    model = tf.keras.models.load_model(name_model,
                                       custom_objects={'loss': cvf.dice_coef_loss},
                                       compile=False)
    input_size = (len(model.layers[0].output_shape[0])) - 1

    return model, input_size


def detect_lumen(model, innput_frame, output_size=(300, 300)):
    # get the input size of the network
    input_layer_shape = model.layers[0].input_shape
    shape_input = np.shape(input_layer_shape)
    if shape_input[-1] == 4:
        input_size_x = input_layer_shape[0][1]
        input_size_y = input_layer_shape[0][2]

    # reshape the input frame to be compatible with the input of the network
    reshaped_img = cv2.resize(innput_frame, (input_size_x, input_size_y),
                              interpolation=cv2.INTER_AREA)
    # apply blur to the image and normalize the image
    resized = (cv2.blur(reshaped_img, (7, 7)))/255
    # make a prediction of the mask
    mask = cvf.predict_mask(model, resized)
    # resize the image for a size to show to the user
    output_imgage = cv2.resize(reshaped_img, output_size, interpolation=cv2.INTER_AREA)
    w, h, d = np.shape(output_imgage)
    # calculate the center points of the lumen according to the detected mask
    point_x, point_y = cvf.detect_dark_region(mask, output_imgage)

    #if not(np.isnan(point_x)):
    #    cv2.circle(output_imgage, (int(point_x), int(point_y)), 45, (0, 0, 255), 2)
    #if not(np.isnan(point_y)):
    #    cv2.circle(output_imgage, (int(point_x), int(point_y)), 25, (0, 0, 255), 2)
    #    cv2.line(output_imgage, (int(point_x), int(point_y)), (int(w / 2), int(h / 2)), (255, 0, 0), 4)
    # center of the image
    #cv2.rectangle(output_imgage, (int(w / 2) - 3, int(h / 2) - 3), (int(w / 2) + 3, int(h / 2) + 3),  (0, 255, 255), -1)

    return output_imgage, point_x, point_y


def main():
    cap = cv2.VideoCapture(0)
    project_folder = '/home/nearlab/Jorge/current_work/lumen_segmentation/data/' \
                     'phantom_lumen/results/ResUnet/'
    folder_name = 'ResUnet_lr_0.0001_bs_16_rgb_19_05_2021_17_07'
    #folder_name = 'ResUnet_lr_0.0001_bs_8_lab_12_12_2020_16_43' # good
    #folder_name = 'ResUnet_lr_0.001_bs_8_hsv_13_12_2020_13_26' # even better

    model, input_size = load_model(project_folder, folder_name)
    frame_rate = 60
    point_x, point_y = 0, 0

    while cap.isOpened():
        prev = 0
        ret, frame = cap.read()
        init_time = time.time()
        time_elapsed = time.time() - prev
        if ret is True:
            if time_elapsed > 1. / frame_rate:
                detected, ptx, pty = detect_lumen(model, frame)
                cv2.imshow('frame', detected)
                print('detected point:', ptx, pty, 'frequency:', 1/(time.time()-init_time))
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        else:
            break

    # Release everything if job is finished
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()



