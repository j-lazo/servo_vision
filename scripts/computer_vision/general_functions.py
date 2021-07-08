import cv2
import numpy as np
import copy
import keras.backend as K
from scipy.linalg import expm, inv
from numpy import dot, eye
import os


def paint_image(painted_image, center_point_x, center_point_y,
                radius_center_point=10, radius_delta_1=25, radius_delta_2=45):
    h, w, d = np.shape(painted_image)
    cv2.circle(painted_image, (int(center_point_x), int(center_point_y)), radius_center_point, (0, 0, 255), -1)
    cv2.circle(painted_image, (int(center_point_x), int(center_point_y)), radius_delta_1, (0, 0, 255), 2)
    cv2.circle(painted_image, (int(center_point_x), int(center_point_y)), radius_delta_2, (0, 0, 255), 2)
    cv2.line(painted_image, (int(center_point_x), int(center_point_y)), (int(h / 2), int(w / 2)), (255, 0, 0), 4)

    return painted_image


def build_input_network(img_modality, ):

    def read_image_test(path, img_modality='rgb'):
        if img_modality == 'npy':
            x = np.load(path, allow_pickle=True)
            x = np.resize(x, (3, 256, 256, 3))
            x = x / 255.0

        elif img_modality == 'ensemble':
            x_vol = np.load(path, allow_pickle=True)
            x_vol = np.resize(x_vol, (3, 256, 256, 3))
            x_vol = x_vol / 255.0
            x_frame = x_vol[0]
            x_frame = cv2.resize(x_frame, (256, 256))
            x_frame = x_frame / 255.0
            x = x_vol, x_frame

        else:
            x = cv2.imread(path, cv2.IMREAD_COLOR)
            x = cv2.resize(x, (256, 256))
            x = x / 255.0

    return input_network


def calculate_middle_point_chessboard(corners):

    """
    Caclulates the middle point of a chessboard given the corners of the points
    @param corners: the corners detected
    @return: the point x,y of the chessboard
    """

    points_x = []
    points_y = []

    for point in corners:
        points_x.append(point[0][0])
        points_y.append(point[0][1])

    point_x = int(np.mean(points_x))
    point_y = int(np.mean(points_y))

    return point_x, point_y


def detect_corners_chessboard(frame):

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    found, corners = find_corners(gray)
    if found:
        output_image = draw_corners(gray, corners)
        point_x, point_y = calculate_middle_point_chessboard(corners)

    else:
        output_image = gray
        point_x = np.nan
        point_y = np.nan

    return output_image, point_x, point_y


def find_corners(image, pattern_size=(7, 5)):
    """
    @param image: grayscale image
    @param pattern_size:
    @return:
    """
    found, corners = cv2.findChessboardCorners(image, pattern_size)
    term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.1)
    if found:
        cv2.cornerSubPix(image, corners, (5, 5), (-1, -1), term)
    return found, corners


def draw_corners(image, corners, pattern_size=(7, 5)):
    color_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    cv2.drawChessboardCorners(color_image, pattern_size, corners, True)
    return color_image


def get_object_pose(object_points, image_points, camera_matrix, dist_coeffs):
    ret, rvec, tvec = cv2.solvePnP(object_points, image_points, camera_matrix, dist_coeffs)
    return rvec.flatten(), tvec.flatten()


def calibrate_lens(dir_imgs, pattern_size, square_size):
    image_list = sorted(os.listdir(dir_imgs))
    img_points, obj_points = [], []
    h,w = 0, 0
    pattern_points = np.zeros((np.prod(pattern_size), 3), np.float32)
    pattern_points[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)
    pattern_points *= square_size

    for image_name in image_list:

        img = cv2.imread(dir_imgs + image_name)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = img.shape[:2]
        found, corners = find_corners(img, pattern_size)

        if not found:
            raise Exception("chessboard calibrate_lens Failed to find corners in img")
        img_points.append(corners.reshape(-1, 2))
        obj_points.append(pattern_points)

    camera_matrix = np.zeros((3, 3))
    dist_coeffs = np.zeros(5)
    #rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, (w,h))
    cv2.calibrateCamera(obj_points, img_points, (w, h), camera_matrix, dist_coeffs)
    return camera_matrix, dist_coeffs


def hat(v):
    return [[0,    -v[2],  v[1]],
            [v[2],     0, -v[0]],
            [-v[1],  v[0],    0]]


def tf_mat(r, t):
    res = eye(4)
    res[0:3, 0:3] = expm(hat(r))
    res[0:3, -1] = t
    return res


def detect_dark_region(mask, image):
    """
    Caclualtes the middle points of the mask depending on...
    @param mask:
    @param image:
    @return:
    """

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


def clean_mask(mask, threshold=0.15):

    """

    @param mask:
    @param threshold: a percentage to be compared
    @return:
    """

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
            # if an area is (threshold %)smaller than the bigger area, mark to remove
            if sorted_areas[i+1] < threshold * sorted_areas[0]:
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


def calc_histograms_and_center(mask, image, method='direct_mask'):

    point_x = np.nan
    point_y = np.nan

    if method == 'direct_mask':
        # calculates the centroid of the lumen using directly the mask
        if not (np.all(mask == 0)):
            mask = np.array(mask * 255, dtype=np.uint8)
            color = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)  # convert to grayscale
            blur = cv2.blur(gray, (3, 3))  # blur the image
            ret, thresh = cv2.threshold(blur, 90, 255, cv2.THRESH_BINARY)

            # Finding contours for the thresholded image
            contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            # create hull array for convex hull points
            new_hulls = []
            new_contours = []
            # calculate points for each contour

            if len(contours) > 1:
                temp_contours = contours[0]
                for i in range(1, len(contours)):
                    temp_contours = np.concatenate((temp_contours, contours[i]), axis=0)

                new_contours.append(temp_contours)
            else:
                new_contours = contours

            for i in range(len(new_contours)):
                # print('new_contour', type(new_contours[i]))
                # print(new_contours[i])
                new_hulls.append(cv2.convexHull(new_contours[i], False))
            if new_contours:
                M = cv2.moments(new_contours[0])
                if M["m00"] != 0:
                    point_x = int(M["m10"] / M["m00"])
                    point_y = int(M["m01"] / M["m00"])
                else:
                    point_x = np.nan
                    point_y = np.nan
            else:
                point_x = np.nan
                point_y = np.nan

        else:
            point_x = np.nan
            point_y = np.nan

    elif method == 'dark_points':

        # calculates the centroid of considering only the dark points above certain threshold
        if not (np.all(mask == 0)):
            percentage = 0.6
            # grayscale = cv2.cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            grayscale = image[:, :, 2]
            grayscale = np.multiply(grayscale, mask)

            # list_values_grayscale = [value for row in grayscale for value in row if value != 0]
            # create the histogram plot, with three lines, one for
            # each color
            max_grays = ((np.where(grayscale >= int(percentage * np.amax(grayscale)))))
            # in this case the center is calculated using only the grayscale image
            gray_contours = np.asarray([build_contours(max_grays)])
            gray_convex_hull, point_x, point_y = determine_convex_hull(gray_contours)

            points_x = []
            points_y = []
            for hull in gray_convex_hull:
                for i, point in enumerate(hull):
                    points_x.append(point[0][0])
                    points_y.append(point[0][1])

        else:
            point_x = np.nan
            point_y = np.nan

    return point_x, point_y