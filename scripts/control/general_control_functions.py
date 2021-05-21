import numpy as np


def naive_control(current_x, current_y, current_z, target_x, target_y, absolute_delta, user_define_step=0.01):
    target_vector = [0, 0, 0]
    print(current_x, current_y, target_x, target_x)
    if target_x ** 2 + target_y ** 2 > absolute_delta ** 2:
        target_vector[0] = current_x + target_x * user_define_step
        # target_vector[1] = current_y + target_y * user_define_step
        # target_vector[2] = current_z + delta_z * user_define_step

    else:
        target_vector = 0
        target_vector[2] = 2
    print(target_vector)
    return target_vector
