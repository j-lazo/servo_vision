import time
import numpy as np
import cv2
from glob import glob
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger, TensorBoard
from tqdm import tqdm
import tensorflow as tf
import keras.backend as K
from tensorflow.keras.backend import sum as suma
from tensorflow.keras.backend import mean
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from keras.utils import CustomObjectScope

import copy
from os import listdir
from os.path import isfile, join
from datetime import datetime
import csv
import matplotlib.pyplot as plt

from sklearn.metrics import average_precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score

image_modality = 'rgb'

augmented = True

if augmented is True:
    amount_data = '/augmented_data/'
else:
    amount_data = '/original_data/'

analyze_validation_set = False
evaluate_train_dir = False


def load_data(path):
    print(path)
    path_images = ''.join([path, 'image/', image_modality, "/*"])
    path_labels = ''.join([path, "label/*"])
    images = sorted(glob(path_images))
    masks = sorted(glob(path_labels))
    total_size_images = len(images)
    total_size_labels = len(masks)
    print('total size images:', total_size_images, path_images)
    print('total size labels:', total_size_labels, path_labels)
    return (images, masks)

def load_data_only_imgs(path):
    print(path)
    path_images = ''.join([path, "/*"])
    images = sorted(glob(path_images))
    total_size_images = len(images)
    print('total size images:', total_size_images, path_images)
    return (images, images)


def read_image_test(path):
    x = cv2.imread(path, cv2.IMREAD_COLOR)
    x = cv2.resize(x, (256, 256))
    x = x / 255.0
    return x


def read_mask_test(path):
    x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    x = cv2.resize(x, (256, 256))
    x = np.expand_dims(x, axis=-1)
    return x


def read_image(path):

    path = path.decode()
    x = cv2.imread(path, 1)
    x = cv2.resize(x, (256, 256))
    x = x/255.0
    return x


def read_mask(path):
    path = path.decode()
    x = cv2.imread(path)
    x = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
    x = cv2.resize(x, (256, 256))
    x = x/255.0
    x = np.expand_dims(x, axis=-1)
    return x


def tf_parse(x, y):
    def _parse(x, y):
        x = read_image(x)
        y = read_mask(y)
        return x, y

    x, y = tf.numpy_function(_parse, [x, y], [tf.float64, tf.float64])
    x.set_shape([256, 256, 3])
    y.set_shape([256, 256, 1])
    return x, y


def tf_dataset(x, y, batch=8):
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    dataset = dataset.map(tf_parse)
    dataset = dataset.batch(batch)
    dataset = dataset.repeat()
    return dataset


def iou(y_true, y_pred, smooth=1e-15):
    def f(y_true, y_pred):
        intersection = (y_true * y_pred).sum()
        union = y_true.sum() + y_pred.sum() - intersection
        x = (intersection + smooth) / (union + smooth)
        x = x.astype(np.float32)
        return x

    return tf.numpy_function(f, [y_true, y_pred], tf.float32)




def dice_coef(y_true, y_pred, smooth=1):
    intersection = K.sum(y_true * y_pred, axis=[1, 2, 3])
    union = K.sum(y_true, axis=[1, 2, 3]) + K.sum(y_pred, axis=[1, 2, 3])
    return K.mean((2. * intersection + smooth) / (union + smooth), axis=0)


def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)


def conv_block(x, num_filters):
    x = Conv2D(num_filters, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    skip = Conv2D(num_filters, (3, 3), padding="same")(x)
    skip = Activation("relu")(skip)
    skip = BatchNormalization()(skip)

    x = Conv2D(num_filters, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = tf.math.add_n([x, skip])
    x = Activation("relu")(x)

    return x


def build_model():
    size = 256
    num_filters = [16, 32, 48, 64]
    # num_filters = [64, 48, 32, 16]
    # num_filters = [64, 128, 256, 512]
    inputs = Input((3, size, size, 3))
    skip_x = []
    x = inputs
    for f in num_filters:
        x = conv_block(x, f)
        print(str(x.shape.as_list()))
        skip_x.append(x)
        x = MaxPool2D((2, 2))(x)

    ## Bridge
    x = conv_block(x, num_filters[-1])

    num_filters.reverse()
    skip_x.reverse()
    ## Decoder
    for i, f in enumerate(num_filters):
        x = UpSampling2D((2, 2))(x)
        xs = skip_x[i]
        x = Concatenate()([x, xs])
        x = conv_block(x, f)

    ## Output
    x = Conv2D(1, (1, 1), padding="same")(x)
    x = Activation("sigmoid")(x)

    return Model(inputs, x)


def mask_parse(mask):
    mask = np.squeeze(mask)
    mask = [mask, mask, mask]
    mask = np.transpose(mask, (1, 2, 0))
    return mask


def read_image_test(path):
    x = cv2.imread(path, cv2.IMREAD_COLOR)
    x = cv2.resize(x, (256, 256))
    x = x / 255.0
    return x


def read_mask_test(path):
    x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    x = cv2.resize(x, (256, 256))
    x = np.expand_dims(x, axis=-1)
    return x



def dice(im1, im2, smooth=0.001):
    im1 = np.asarray(im1).astype(np.bool)
    im2 = np.asarray(im2).astype(np.bool)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    # Compute Dice coefficient
    intersection = np.logical_and(im1, im2)
    if (np.unique(im1) == [False]).all() and (np.unique(im2) == [False]).all():
        dsc = 1.
    else:
        dsc = 2. * (intersection.sum() + smooth) / (im1.sum() + im2.sum() + smooth)

    return dsc
    # return 2. * (intersection.sum() + smooth) / (im1.sum() + im2.sum() + smooth)


def read_img(dir_image):
    original_img = cv2.imread(dir_image)
    img = cv2.resize(original_img, (256, 256))
    img = img / 255
    return img


def predict_mask(model, input_image):
    return model.predict(np.expand_dims(input_image, axis=0))[0] >= 0.5


def paint_imgs(img, mask):

    if np.shape(img) != np.shape(mask):
        img = cv2.resize(img, (np.shape(mask)[0], np.shape(mask)[1]))

    for i in range(np.shape(mask)[0]):
        for j in range(np.shape(mask)[1]):
            if mask[i, j, 0] == True:
                img[i, j, 1] = 100

    return img


def build_contours(array_of_points):

    contours = []
    for i, y_points in enumerate(array_of_points[0]):
        point = (array_of_points[1][i], y_points)
        point = np.asarray(point)
        contours.append([point])

    return contours


def calc_histograms_and_center(mask, image):

    if not (np.all(mask == 0)):

        percentage = 0.6
        #grayscale = cv2.cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        grayscale = image[:, :, 2]
        grayscale = np.multiply(grayscale, mask)

        #list_values_grayscale = [value for row in grayscale for value in row if value != 0]
        # create the histogram plot, with three lines, one for
        # each color
        max_grays = ((np.where(grayscale >= int(percentage * np.amax(grayscale)))))
        gray_contours = np.asarray([build_contours(max_grays)])
        gray_convex_hull, gray_x, gray_y = determine_convex_hull(gray_contours)

        points_x = []
        points_y = []
        for hull in gray_convex_hull:
            for i, point in enumerate(hull):
                points_x.append(point[0][0])
                points_y.append(point[0][1])

    else:
        gray_x = 'nAN'
        gray_y = 'nAN'

    return gray_x, gray_y


def detect_dark_region(mask, image):
    w_image, h_image, d_image = np.shape(image)
    w_mask, h_mask, d_mask = np.shape(mask)
    temp_mask = np.zeros((w_mask, h_mask, 3), dtype="float32")
    temp_mask[:, :, 0] = mask[:, :, 0]
    temp_mask[:, :, 1] = mask[:, :, 0]
    temp_mask[:, :, 2] = mask[:, :, 0]

    if w_image != w_mask or h_image != h_mask:
       mask = cv2.resize(temp_mask, (w_image, h_image))

    gt_mask = np.zeros((w_image, h_image))

    if np.any(mask):

        #temp_mask = cv2.cv2.cvtColor(temp_mask*1, cv2.COLOR_BGR2GRAY)
        cl_mask = clean_mask(mask*1)
        gt_mask[cl_mask == 1.0] = 1

    point_x, point_y = calc_histograms_and_center(gt_mask, image)

    return point_x, point_y


def clean_mask(mask):

    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    u8 = mask.astype(np.uint8)
    # remove the small areas appearing in the image if there is a considerable big one
    areas = []
    remove_small_areas = False
    contours, hierarchy = cv2.findContours(u8,
                                           cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        areas.append(cv2.contourArea(contour))

    if len(contours) > 1:
        # sort the areas from bigger to smaller
        sorted_areas = sorted(areas, reverse=True)
        index_remove = np.ones(len(areas))
        for i in range(len(sorted_areas)-1):
            # if an area is 1/4 smaller than the bigger area, mark to remove
            if sorted_areas[i+1] < 0.15 * sorted_areas[0]:
                index_remove[areas.index(sorted_areas[i+1])] = 0
                remove_small_areas = True

    if remove_small_areas is True:
        #new_mask = np.zeros((w, d))
        new_mask = copy.copy(mask)
        for index, remove in enumerate(index_remove):
            if remove == 0:
                # replace the small areas with 0
                cv2.drawContours(new_mask, contours, index, (0, 0, 0), -1)  # as opencv stores in BGR format
    else:
        new_mask = mask

    return new_mask

def determine_convex_hull(contours):

    point_sets = np.asarray(contours[0])
    #contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # create hull array for convex hull points
    new_hulls = []
    new_contours = []
    # calculate points for each contour
    if len(point_sets) > 1:
        temp_contours = point_sets[0]
        for i in range(1, len(point_sets)):
            temp_contours = np.concatenate((temp_contours, point_sets[i]), axis=0)

        new_contours.append(temp_contours)
    else:
        new_contours = point_sets

    for i in range(len(new_contours)):
        new_hulls.append(cv2.convexHull(new_contours[i], False))

    if new_hulls == []:
        point_x = 'old'
        point_y = 'old'
    else:
        M = cv2.moments(new_hulls[0])
        if M["m00"] < 0.001:
            M["m00"] = 0.001

        point_x = int(M["m10"] / M["m00"])
        point_y = int(M["m01"] / M["m00"])

    return new_hulls, point_x, point_y

def main():
    cap = cv2.VideoCapture(0)
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.avi', fourcc, 20.0, (300, 300))

    # --------------Make predictions -------------
    lr = lr = 1e-3
    opt = tf.keras.optimizers.Adam(lr)
    metrics = ["acc", tf.keras.metrics.Recall(),
               tf.keras.metrics.Precision(),
               dice_coef,
               iou]

    project_folder = '/home/nearlab/Jorge/current_work/' \
                     'lumen_segmentation/data/lumen_data/'
    #folder_name = 'ResUnet_lr_0.0001_bs_8_lab_12_12_2020_16_43' # good
    folder_name = 'ResUnet_lr_0.001_bs_8_hsv_13_12_2020_13_26' # even better
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
                                       custom_objects={'loss': dice_coef_loss},
                                       compile=False)

    model.compile(optimizer=opt, loss=dice_coef_loss, metrics=metrics)
    #model.summary()
    frame_rate = 15
    point_x, point_y = 0, 0
    points_x = []
    points_y = []
    while (cap.isOpened()):
        prev = 0
        ret, frame = cap.read()


        init_time = time.time()
        time_elapsed = time.time() - prev
        if ret == True:
            #frame = cv2.flip(frame, 0)
            #out.write(frame)
            if time_elapsed > 1. / frame_rate:
                reshaped = cv2.resize(frame, (256, 256), interpolation=cv2.INTER_AREA)
                reshaped = cv2.blur(reshaped, (7, 7))
                #img_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                mask = predict_mask(model, reshaped)
                #painted_frame = paint_imgs(resized, mask)
                resized_2 = cv2.resize(reshaped, (300, 300), interpolation=cv2.INTER_AREA)
                w, h, d = np.shape(resized_2)
                previous_point_x = point_x
                previous_point_y = point_y
                point_x, point_y = detect_dark_region(mask, resized_2)
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
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()



