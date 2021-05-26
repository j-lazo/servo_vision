import math
import numpy as np


def tarnsform_to_img_space(point_x, point_y, shape_img):
    cx = shape_img[0] / 2
    cy = shape_img[1] / 2
    corrected_x = point_x - cx
    corrected_y = -(point_y - cy)
    return corrected_x, corrected_y


def naive_control(current_x, current_y, current_z, target_x, target_y,
                  img_shape, absolute_delta, user_define_step=0.003):
    tranformed_x, tranformed_y = tarnsform_to_img_space(target_x, target_y, img_shape)
    tranformed_x = tranformed_x * -1
    target_vector = [current_x, current_y, current_z]

    if (tranformed_x ** 2 + tranformed_y ** 2) > absolute_delta ** 2:
        print('updating x, y')
        target_vector[0] = current_x + tranformed_x * user_define_step
        target_vector[1] = current_y + tranformed_y * user_define_step
        target_vector[2] = current_z
        # target_vector[2] = current_z + delta_z * user_define_step

    else:
        print('forward')
        target_vector[0] = current_x
        target_vector[1] = current_y
        target_vector[2] = current_z + 0.5

    return target_vector


def less_naive_control(current_z, target_x, target_y,
                       img_shape, absolute_delta, user_define_step=0.003):
    transformed_x, transformed_y = tarnsform_to_img_space(target_x, target_y, img_shape)
    transformed_x = transformed_x * -1
    target_distance = math.sqrt(transformed_x ** 2 + transformed_y ** 2)
    target_vector = [0, 0, current_z]

    if target_distance > absolute_delta:
        print('updating x, y')
        print(target_distance)  # for later fine tune the number
        target_vector[0] = transformed_x * user_define_step
        target_vector[1] = transformed_y * user_define_step
        target_vector[2] = current_z
        # target_vector[2] = current_z + delta_z * user_define_step

    elif target_distance < absolute_delta & target_distance > absolute_delta * 0.2:
        target_vector[0] = transformed_x * user_define_step * (target_distance / absolute_delta)
        target_vector[1] = transformed_y * user_define_step * (target_distance / absolute_delta)
        target_vector[2] = current_z + target_distance / absolute_delta

    else:
        print('forward')
        target_vector[0] = 0
        target_vector[1] = 0
        target_vector[2] = current_z + 0.2

    return target_vector


def jocobian_correcion_control(new_jacobian, target_x, target_y, img_shape, absolute_delta, user_define_step=1):
    transformed_x, transformed_y = tarnsform_to_img_space(target_x, target_y, img_shape)
    transformed_x = transformed_x * -1
    target_distance = math.sqrt(transformed_x ** 2 + transformed_y ** 2)

    target_vector = np.array([[transformed_x], [transformed_y]])
    inverse_jacobian = np.linalg.inv(new_jacobian())
    actuate_vector = np.matul(inverse_jacobian, target_vector)

    if target_distance > absolute_delta:
        actuate_vector_z = [0]
    else:
        actuate_vector_z = [1]

    return actuate_vector + actuate_vector_z


def update_jacobian(current_jacobian, delta_q, delta_actual_displacement, beta):
    delta_q = np.array(delta_q)
    delta_actual_displacement = np.array(delta_actual_displacement)

    new_jacobian = current_jacobian + beta * (
                delta_actual_displacement - current_jacobian * delta_q) * delta_q.transpose \
                   / (delta_q.transpose() * delta_q)

    return new_jacobian
