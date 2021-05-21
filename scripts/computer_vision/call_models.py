import time
import numpy as np
import cv2
from tensorflow.keras.layers import *
import tensorflow as tf
import general_functions as cvf


def main():
    cap = cv2.VideoCapture(0)
    #project_folder = '/home/nearlab/Jorge/current_work/' \
    #                 'lumen_segmentation/data/lumen_data/'
    project_folder = '/home/nearlab/Jorge/current_work/lumen_segmentation/data/' \
                     'phantom_lumen/'
    folder_name = 'ResUnet_lr_0.0001_bs_16_rgb_19_05_2021_17_07'
    #folder_name = 'ResUnet_lr_0.0001_bs_8_lab_12_12_2020_16_43' # good
    #folder_name = 'ResUnet_lr_0.001_bs_8_hsv_13_12_2020_13_26' # even better
    #folder_name = 'ResUnet_lr_1e-05_bs_8_rgb_13_04_2021_19_59'
    #folder_name = 'ResUnet_lr_1e-05_bs_16_rgb_27_04_2021_20_10'
    #folder_name = 'ResUnet_lr_0.0001_bs_8_hsv_28_04_2021_19_34'

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

    while cap.isOpened():
        prev = 0
        ret, frame = cap.read()
        init_time = time.time()
        time_elapsed = time.time() - prev
        if ret == True:
            #frame = cv2.flip(frame, 0)
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


if __name__ == "__main__":
    main()



