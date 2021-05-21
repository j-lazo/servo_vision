import cv2
import numpy as np
import copy
import keras.backend as K


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

def dice_coef(y_true, y_pred, smooth=1):
    intersection = K.sum(y_true * y_pred, axis=[1, 2, 3])
    union = K.sum(y_true, axis=[1, 2, 3]) + K.sum(y_pred, axis=[1, 2, 3])
    return K.mean((2. * intersection + smooth) / (union + smooth), axis=0)


def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)


def predict_mask(model, input_image):
    return model.predict(np.expand_dims(input_image, axis=0))[0] >= 0.5


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