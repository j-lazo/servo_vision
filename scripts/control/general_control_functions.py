import numpy as np


def naive_control(current_x, current_y, current_z, delta_x, delta_y, absolute_delta):

    displacement_vector = [0, 0, 0]
    rad = np.sqrt(delta_x**2 + delta_y**2)

    if rad < absolute_delta:
        print('True')
        displacement_vector = [0, 0, 1]
    else:
        displacement_vector = [0, 0, -1]

    return displacement_vector