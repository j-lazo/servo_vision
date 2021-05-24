def tarnsform_to_img_space(point_x, point_y, shape_img):
    cx = shape_img[0]/2
    cy = shape_img[1]/2
    corrected_x = point_x - cx
    corrected_y = -(point_y - cy)
    return corrected_x, corrected_y


def naive_control(current_x, current_y, current_z, target_x, target_y,
                  img_shape, absolute_delta, user_define_step=0.003):
    tranformed_x, tranformed_y = tarnsform_to_img_space(target_x, target_y, img_shape)
    #current_x, current_y = tarnsform_to_img_space(current_x, current_y, img_shape)
    tranformed_x = tranformed_x * -1
    #current_x = current_x * -1
    target_vector = [current_x, current_y, current_z]

    if (tranformed_x ** 2 + tranformed_y ** 2) > absolute_delta ** 2:
        target_vector[0] = current_x + tranformed_x * user_define_step
        target_vector[1] = current_y + tranformed_y * user_define_step
        target_vector[2] = 0
        # target_vector[2] = current_z + delta_z * user_define_step

    else:
        target_vector[2] = 0

    return target_vector
