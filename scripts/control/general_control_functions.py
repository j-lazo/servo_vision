import math
import numpy as np


def transform_to_img_space(point_x, point_y, shape_img):
    cx = shape_img[0] / 2
    cy = shape_img[1] / 2
    corrected_x = point_x - cx
    corrected_y = -(point_y - cy)
    return corrected_x, corrected_y


def naive_control(current_x, current_y, current_z, target_x, target_y,
                  img_shape, absolute_delta, user_define_step=0.003):
    tranformed_x, tranformed_y = transform_to_img_space(target_x, target_y, img_shape)
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
    transformed_x, transformed_y = transform_to_img_space(target_x, target_y, img_shape)
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
    transformed_x, transformed_y = transform_to_img_space(target_x, target_y, img_shape)
    transformed_x = transformed_x * -1
    target_distance = math.sqrt(transformed_x ** 2 + transformed_y ** 2)
    target_vector = np.array([[transformed_x], [transformed_y]])
    inverse_jacobian = np.linalg.inv(new_jacobian)
    actuate_vector = np.matul(inverse_jacobian, target_vector).tolist()  # make it back a list.

    if target_distance > absolute_delta:
        actuate_vector_z = [0]
    else:
        actuate_vector_z = [1]
    return actuate_vector + actuate_vector_z  # Here we return a list!


def jacobian_correction_velocity_control(new_jacobian, target_x, target_y, img_shape, absolute_delta,
                                        user_define_step=1, delta_time=34):
    transformed_x, transformed_y = transform_to_img_space(target_x, target_y, img_shape)
    transformed_x = transformed_x * -1
    print(transformed_x)
    print(transformed_y)
    target_distance = math.sqrt(transformed_x ** 2 + transformed_y ** 2)
    target_vector = np.array([transformed_x, transformed_y])
    inverse_jacobian = np.linalg.inv(new_jacobian)
    actuate_vector = (np.matmul(inverse_jacobian, target_vector)/delta_time*2).tolist()  # make it back a list.
    ############VERY IMPORTANT TIME FACTOR CONSIDERATION::: ARDUINO OR PYTHON TIME SYNC ###################
    if target_distance > absolute_delta:
        actuate_vector_z = [0]
    else:
        actuate_vector_z = [1]
    return actuate_vector + actuate_vector_z  # Here we return a list!


def update_jacobian(current_jacobian, delta_time, delta_q, point_x, point_y, previous_point_x, previous_point_y,
                    beta=0.05):
    delta_q = np.array(delta_q).astype(float) / delta_time
    delta_actual_displacement = np.array([point_x - previous_point_x, point_y - previous_point_y]).astype(float) / delta_time
    if delta_q[0] == 0 and delta_q[1] == 0:
        new_jacobian = current_jacobian
    else:
        print ("current_jacobian:", current_jacobian)
        print ("delta_actual_displacement:", delta_actual_displacement)
        print ("delta_q:", delta_q)
        print ("JkQk:", np.matmul(current_jacobian, delta_q))
        print ("x-JkQk:", delta_actual_displacement - np.matmul(current_jacobian, delta_q))
        print ("(x-JkQk)*Qk^T: ", np.outer(delta_actual_displacement - np.matmul(current_jacobian, delta_q), delta_q))
        print ("Qk^T:", np.array([delta_q]).transpose(), np.array([delta_q]).transpose()[0])
        new_jacobian = current_jacobian + beta * np.outer(
            delta_actual_displacement - np.matmul(current_jacobian, delta_q), np.array(delta_q)) / np.dot(delta_q, delta_q)

    return new_jacobian


def main():
    jacobian = np.array([[0.625, -0.18], [0.08, 1.0]])
    delta_q = [1, 0]
    point_x, point_y, previous_point_x, previous_point_y = 1.0, 0.0, 3.0, 0.0
    target_x, target_y = 2.0, 3.0
    new_jacobian = update_jacobian(jacobian, 0.0016, delta_q, point_x, point_y, previous_point_x, previous_point_y, beta=1)
    print ("old jacobian matrix:", jacobian, "and the new one!:", new_jacobian)
    print ("actuation velocity: ", jocobian_correcion_velocity_control(jacobian, target_x, target_y, (400, 600), 30))
    return 0


if __name__ == "__main__":
    main()
