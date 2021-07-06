import math
import numpy as np
import random
import time


def transform_to_img_space(point_x, point_y, shape_img):
    cx = shape_img[0] / 2
    cy = shape_img[1] / 2
    corrected_x = point_x - cx
    corrected_y = -(point_y - cy)
    return corrected_x, corrected_y


def naive_control(current_x, current_y, current_z, target_x, target_y,
                  img_shape, absolute_delta, user_define_step=0.003):
    transformed_x, transformed_y = transform_to_img_space(target_x, target_y, img_shape)
    transformed_x = transformed_x * -1
    target_vector = [current_x, current_y, current_z]

    if (transformed_x ** 2 + transformed_y ** 2) > absolute_delta ** 2:
        print('updating x, y')
        target_vector[0] = current_x + transformed_x * user_define_step
        target_vector[1] = current_y + transformed_y * user_define_step
        target_vector[2] = current_z
        # target_vector[2] = current_z + delta_z * user_define_step

    else:
        print('forward')
        target_vector[0] = current_x
        target_vector[1] = current_y
        target_vector[2] = current_z + 0.5

    return target_vector


def less_naive_control(current_z, target_x, target_y,
                       img_shape, absolute_delta, propotional_gain=0.5):
    transformed_x, transformed_y = transform_to_img_space(target_x, target_y, img_shape)
    transformed_x = transformed_x * -1 * propotional_gain
    transformed_y = transformed_y * propotional_gain
    target_distance = math.sqrt(transformed_x ** 2 + transformed_y ** 2)
    target_vector = [0, 0, current_z]

    if target_distance > absolute_delta:
        print('updating x, y')
        print(target_distance)  # for later fine tune the number
        # target_vector[0] = transformed_x * user_define_step
        # target_vector[1] = transformed_y * user_define_step
        target_vector[0] = (transformed_x * 1 + transformed_y * 0.18)
        target_vector[1] = (transformed_x * -0.01 + transformed_y * 0.625)
        target_vector[2] = current_z
        # target_vector[2] = current_z + delta_z * user_define_step

    elif target_distance < absolute_delta & target_distance > absolute_delta * 0.2:
        # target_vector[0] = transformed_x * user_define_step * (target_distance / absolute_delta)
        # target_vector[1] = transformed_y * user_define_step * (target_distance / absolute_delta)
        target_vector[0] = (transformed_x * 1 + transformed_y * 0.18) / 0.6268 * 0.05 * (
                target_distance / absolute_delta)
        target_vector[1] = transformed_x * -0.01 + transformed_y * 0.625 / 0.6268 * 0.05 * (
                target_distance / absolute_delta)
        target_vector[2] = current_z + target_distance / absolute_delta

    else:
        print('forward')
        target_vector[0] = 0
        target_vector[1] = 0
        target_vector[2] = current_z + 0.2

    return target_vector


def nasty_control(target_x, target_y,
                  img_shape):
    # initialize the parameters
    if 0 < target_x < img_shape[0] and 0 < target_y < img_shape[1]:

        transformed_x, transformed_y = transform_to_img_space(target_x, target_y, img_shape)
        transformed_x = transformed_x * -1
        transformed_y = transformed_y
        target_distance = math.sqrt(transformed_x ** 2 + transformed_y ** 2)
    else:
        print("target out of boundary")
        return [0.0, 0.0], 0.0, 0.0

    # mapping the right zone
    unit_vector_x, unit_vector_y, theta = mapping_direction(transformed_x, transformed_y)
    magnitude = mapping_distance(target_distance)
    target_vector = [unit_vector_x * magnitude, unit_vector_y * magnitude]

    return target_vector, theta, magnitude


def discrete_delay_control(target_x, target_y, img_shape, absolute_delta=30, p_gain=1.0):
    if 0 < target_x < img_shape[0] and 0 < target_y < img_shape[1]:
        transformed_x, transformed_y = transform_to_img_space(target_x, target_y, img_shape)
        transformed_x = transformed_x * -1
        target_distance = math.sqrt(transformed_x ** 2 + transformed_y ** 2)
    else:
        print("target out of boundary")
        return [0.0, 0.0], 0.0, 0.0
    target_vector = np.array([transformed_x, transformed_y])
    #    inverse_jacobian = np.linalg.inv(new_jacobian)
    #    actuate_vector = np.matul(inverse_jacobian, target_vector).tolist()  # make it back a list.

    magnitude = mapping_distance(target_distance)
    p_gain = magnitude / 30.0
    theta = math.atan2(float(transformed_y), float(transformed_x))
    unit_vector = [np.cos(theta), np.sin(theta)]
    target_vector = [transformed_x * p_gain, transformed_y * p_gain]

    return target_vector, theta, magnitude  # Here we return a list!


def discrete_jacobian_control(target_x, target_y, img_shape, absolute_delta=30, p_gain=1.0):
    if 0 < target_x < img_shape[0] and 0 < target_y < img_shape[1]:
        transformed_x = -(target_x - img_shape[0] / 2)
        transformed_y = -(target_y - img_shape[1] / 2)
        target_distance = math.sqrt(transformed_x ** 2 + transformed_y ** 2)
    else:
        print("target out of boundary")
        return [0.0, 0.0], 0.0, 0.0
    inv_jacobian = [[0.81, 0.26], [-0.037, 0.33]]
    target_vector = [0, 0]
    magnitude = mapping_distance(target_distance)
    p_gain = magnitude / 30.0
    theta = math.atan2(float(transformed_y), float(transformed_x))
    unit_vector = [np.cos(theta), np.sin(theta)]
    target_vector[0] = p_gain * (inv_jacobian[0][0] * transformed_x + inv_jacobian[0][1] * transformed_y)
    target_vector[1] = p_gain * (inv_jacobian[1][0] * transformed_x + inv_jacobian[1][1] * transformed_y)

    return target_vector, theta, magnitude  # Here we return a list


def update_jacobian_control(jacobian_mat, target_x, target_y, img_shape, absolute_delta=30, p_gain=1.0):
    if 0 < target_x < img_shape[0] and 0 < target_y < img_shape[1]:
        transformed_x = -(target_x - img_shape[0] / 2)
        transformed_y = -(target_y - img_shape[1] / 2)
        target_distance = math.sqrt(transformed_x ** 2 + transformed_y ** 2)
    else:
        print("target out of boundary")
        return [0.0, 0.0], 0.0, 0.0

    inv_jacobian = [[jacobian_mat[1][1], -jacobian_mat[0][1]], [-jacobian_mat[1][0], jacobian_mat[0][0]]]
    target_vector = [0, 0]
    magnitude = mapping_distance(target_distance)
    p_gain = magnitude / 30.0
    theta = math.atan2(float(transformed_y), float(transformed_x))
    unit_vector = [np.cos(theta), np.sin(theta)]
    target_vector[0] = p_gain * (inv_jacobian[0][0] * transformed_x + inv_jacobian[0][1] * transformed_y)
    target_vector[1] = 2.1 * p_gain * (inv_jacobian[1][0] * transformed_x + inv_jacobian[1][1] * transformed_y)

    return target_vector, theta, magnitude  # Here we return a list!


def jacobian_correction_velocity_control(new_jacobian, target_x, target_y, img_shape, absolute_delta,
                                         user_define_step=1, delta_time=34):
    transformed_x, transformed_y = transform_to_img_space(target_x, target_y, img_shape)
    transformed_x = transformed_x * -1
    print(transformed_x)
    print(transformed_y)
    target_distance = math.sqrt(transformed_x ** 2 + transformed_y ** 2)
    target_vector = np.array([transformed_x, transformed_y])
    inverse_jacobian = np.linalg.inv(new_jacobian)
    actuate_vector = (np.matmul(inverse_jacobian, target_vector) / delta_time * 2).tolist()  # make it back a list.
    ############VERY IMPORTANT TIME FACTOR CONSIDERATION::: ARDUINO OR PYTHON TIME SYNC ###################
    if target_distance > absolute_delta:
        actuate_vector_z = [0]
    else:
        actuate_vector_z = [1]
    return actuate_vector + actuate_vector_z  # Here we return a list!


def update_jacobian(current_jacobian, previous_qs, point_x, point_y, previous_point_x, previous_point_y,
                    beta=0.05):
    print(previous_qs)
    zipped_lists = zip(previous_qs[-1], previous_qs[-2])
    delta_q = [x - y for (x, y) in zipped_lists][:2]

    delta_q = np.array(delta_q).astype(float)
    delta_actual_displacement = np.array([point_x - previous_point_x, point_y - previous_point_y]).astype(
        float)
    current_jacobian = np.array(current_jacobian)
    print('delta q', delta_q)
    if delta_q[0] == 0 and delta_q[1] == 0:
        new_jacobian = current_jacobian
    else:
        print("current_jacobian:", current_jacobian)
        print("delta_actual_displacement:", delta_actual_displacement)
        print("delta_q:", delta_q)
        print("JkQk:", np.matmul(current_jacobian, delta_q))
        print("x-JkQk:", delta_actual_displacement - np.matmul(current_jacobian, delta_q))
        print("(x-JkQk)*Qk^T: ", np.outer(delta_actual_displacement - np.matmul(current_jacobian, delta_q), delta_q))
        print("Qk^T:", np.array([delta_q]).transpose(), np.array([delta_q]).transpose()[0])
        new_jacobian = current_jacobian + beta * np.outer(
            delta_actual_displacement - np.matmul(current_jacobian, delta_q), np.array(delta_q)) / np.dot(delta_q,
                                                                                                          delta_q)

    return new_jacobian.tolist()


def jacobian_transformation(transformed_x, transoformed_y):
    return 0


def mapping_direction(transformed_x, transformed_y):
    theta = math.atan2(float(transformed_y), float(transformed_x))
    # print ("input theta: ", math.degrees(theta))
    # print ("x in cartesian:", transformed_x, "y in cartesian:", transformed_y)
    step = np.pi / 8
    if theta < 0:
        theta = theta + 2 * np.pi
    if 0 <= theta < step or step * 15 <= theta < step * 16:
        theta = 0
        case = 1
    elif step <= theta < step * 3:
        theta = (step * 1 + step * 3) / 2
        case = 2
    elif step * 3 <= theta < step * 5:
        theta = (step * 3 + step * 5) / 2
        case = 3
    elif step * 5 <= theta < step * 7:
        theta = (step * 5 + step * 7) / 2
        case = 4
    elif step * 7 <= theta < step * 9:
        theta = (step * 7 + step * 9) / 2
        case = 5
    elif step * 9 <= theta < step * 11:
        theta = (step * 9 + step * 11) / 2
        case = 6
    elif step * 11 <= theta < step * 13:
        theta = (step * 11 + step * 13) / 2
        case = 7
    elif step * 13 <= theta < step * 15:
        theta = (step * 13 + step * 15) / 2
        case = 8
    else:
        case = 0
        print(math.degrees(theta))
        print("fail case")
    # print ("pointing in the direction of:", case)
    # print ("out put theta: ", math.degrees(theta))
    unit_vector_x = np.cos(theta)
    unit_vector_y = np.sin(theta)
    return unit_vector_x, unit_vector_y, theta


def mapping_distance(target_distance):
    # nasty
    if target_distance > 200:
        magnitude = 25
    elif 200 > target_distance > 100:
        magnitude = 15
    elif 100 > target_distance > 30:
        magnitude = 5
    else:
        magnitude = 0
    """# nasty discreete
    if target_distance > 200:
        magnitude = 30
    elif 200 > target_distance > 100:
        magnitude = 10
    elif 100 > target_distance > 30:
        magnitude = 5
    else:
        magnitude = 0"""

    # descrete
    """if target_distance > 150:
        magnitude = 5
    elif 150 >= target_distance > 22:
        magnitude = 6
    else:
        magnitude = 0"""

    # jacobian
    """if target_distance > 80:
        magnitude = 3
    elif 80 >= target_distance > 50:
        magnitude = 4.5
    elif 50 >= target_distance > 22:
        magnitude = 6.5 
    else:
        magnitude = 0"""


    return magnitude


def jacobian_example():
    jacobian = np.array([[0.625, -0.18], [0.08, 1.0]])
    delta_q = [1, 0]
    point_x, point_y, previous_point_x, previous_point_y = 1.0, 0.0, 3.0, 0.0
    target_x, target_y = 2.0, 3.0
    new_jacobian = update_jacobian(jacobian, 0.0016, delta_q, point_x, point_y, previous_point_x, previous_point_y,
                                   beta=1)
    print("old jacobian matrix:", jacobian, "and the new one!:", new_jacobian)
    print("actuation velocity: ", jocobian_correction_velocity_control(jacobian, target_x, target_y, (400, 600), 30))
    return 0


def main():
    while True:
        print(nasty_control(0, 200 - random.randint(-50, 50), 200 - random.randint(-50, 50), (400, 400)))
        time.sleep(0.1)


if __name__ == "__main__":
    main()
