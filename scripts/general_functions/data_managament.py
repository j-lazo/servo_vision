from glob import glob
import numpy as np
import cv2
import tensorflow as tf
import csv
import os
import datetime


def save_data(data_vector, date_experiment):
    results_folder = ''.join([os.getcwd(), '/results/experiment_', date_experiment.strftime("%d_%m_%Y_%H_%M"), '/'])
    if not os.path.isdir(results_folder):
        os.mkdir(results_folder)

    name_test_csv_file = ''.join([results_folder, 'experiment_', date_experiment.strftime("%d_%m_%Y_%H_%M"), '_.csv'])
    print(name_test_csv_file)
    with open(name_test_csv_file, mode='w') as results_file:

        results_file_writer = csv.writer(results_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        results_file_writer.writerow(['vis point x',
                                     'vis point y',
                                     'sensor x',
                                      'sensor y',
                                      'sensor z',
                                      'time',
                                      'upper q',
                                      'side q',
                                      'stepper',
                                      'target x',
                                      'target y',
                                      'theta',
                                      'magnitude',
                                      'actuate signal 1',
                                      'actuate signal 2',
                                      'actuate signal 3'])

        for i in range(np.shape(data_vector)[1]):
            results_file_writer.writerow([data_vector[0][i],
                                          data_vector[1][i],
                                          data_vector[2][i],
                                          data_vector[3][i],
                                          data_vector[4][i],
                                          data_vector[5][i],
                                          data_vector[6][i][0],
                                          data_vector[6][i][1],
                                          data_vector[6][i][2],
                                          data_vector[7][i][0],
                                          data_vector[7][i][1],
                                          data_vector[8][i],
                                          data_vector[9][i],
                                          data_vector[10][i][0],
                                          data_vector[10][i][1],
                                          data_vector[10][i][2]])

    print('Saved at:', name_test_csv_file)


def calculate_average_points(list_of_points):
    clean_list = [point for point in list_of_points if point != np.nan]
    average_point = np.mean(clean_list)
    return average_point

def load_data(path, image_modality):
    """
    @param path: path of the directory with images and masks
    @return: tupple of the

    """
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