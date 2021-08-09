import numpy as np
from matplotlib import pyplot as plt


def get_centers(point_1, point_2, axis):
    if axis == 'x':
        center_point = (point_1[0] - point_2[0])/2
    elif axis == 'y':
        center_point = (point_1[1] - point_2[1])/2
    elif axis == 'z':
        center_point = (point_1[2] - point_2[2])/2

    return center_point


def build_trajectory(set_of_points, trajectory_type):
    """

    @param set_of_points: set of initial 8 points
    @param trajectory_type: straight, right_curve, left_curve, two_curves
    @return:
    """
    # lower square
    poitn_1 = set_of_points[0]
    poitn_2 = set_of_points[1]
    poitn_3 = set_of_points[2]
    poitn_4 = set_of_points[3]

    # upper square
    poitn_5 = set_of_points[4]
    poitn_6 = set_of_points[5]
    poitn_7 = set_of_points[6]
    poitn_8 = set_of_points[7]

    center_y_1 = get_centers(poitn_1, poitn_2, 'y')
    center_y_2 = get_centers(poitn_3, poitn_4, 'y')

    center_z_1 = get_centers(poitn_1, poitn_3, 'z')
    center_z_2 = get_centers(poitn_2, poitn_4, 'z')

    center_x_1 = get_centers(poitn_1, poitn_3, 'x')
    center_x_2 = get_centers(poitn_2, poitn_4, 'x')

    starting_point_x = np.median([poitn_1[0], poitn_3[0], poitn_5[0], poitn_7[0]])
    starting_point_y = np.median([poitn_1[1], poitn_3[1], poitn_5[1], poitn_7[1]])
    starting_point_z = np.median([poitn_1[2], poitn_3[2], poitn_5[2], poitn_7[2]])

    if trajectory_type == 'straight':
        length_segment = 200
        y = np.linspace(starting_point_y, starting_point_y + length_segment, int((starting_point_y + length_segment)/(0.05)) + 1, endpoint=True)
        x = np.ones(len(y)) * np.median([center_x_1, center_x_2])
        z = np.ones(len(y)) * np.median([center_z_1, center_z_2])

    if trajectory_type == 'curve_right':
        # segment 1, straight part
        length_segment_1 = 35
        y_1 = np.linspace(starting_point_y, starting_point_y + length_segment_1, int((starting_point_y + length_segment_1)/(0.1)) + 1, endpoint=True)
        x_1 = np.ones(len(y_1)) * np.median([center_x_1, center_x_2])
        z_1 = np.ones(len(y_1)) * np.median([center_z_1, center_z_2])

        # segment 2, curve
        radius = 60
        curve_angle = 15
        center_circle_z = np.median([center_z_1, center_z_2]) - radius
        center_circle_y = y_1[-1]
        length_segment_2 = radius * np.cos((np.pi/2)-np.deg2rad(curve_angle))
        y_2 = np.linspace(y_1[-1], y_1[-1] + length_segment_2, int((y_1[-1] + length_segment_2)/(0.1)) + 1, endpoint=True)
        x_2 = np.ones(len(y_2)) * np.median([center_x_1, center_x_2])
        z_2 = np.sqrt(radius**2 - (y_2-y_1[-1])**2) + center_circle_z

        # segment 3, second straight part
        straight_lenght = 180
        m = -1/((z_2[-1] - center_circle_z) / (y_2[-1] - center_circle_y))
        b = (z_2[-1]) - (m * (y_2[-1]))
        length_segment_3 = straight_lenght * np.cos(m)

        y_3 = np.linspace(y_2[-1], y_2[-1] + length_segment_3, int((y_2[-1] + length_segment_3)/(0.1)) + 1, endpoint=True)
        x_3 = np.ones(len(y_3)) * np.median([center_x_1, center_x_2])
        z_3 = (m * y_3) + b

        x = np.concatenate((x_1, x_2, x_3))
        y = np.concatenate((y_1, y_2, y_3))
        z = np.concatenate((z_1, z_2, z_3))

    if trajectory_type == 'curve_left':
        # segment 1, straight part
        length_segment_1 = 35
        y_1 = np.linspace(starting_point_y, starting_point_y + length_segment_1,
                          int((starting_point_y + length_segment_1) / (0.1)) + 1, endpoint=True)
        x_1 = np.ones(len(y_1)) * np.median([center_x_1, center_x_2])
        z_1 = np.ones(len(y_1)) * np.median([center_z_1, center_z_2])

        # segment 2, curve
        radius = 60
        curve_angle = 15
        center_circle_z = np.median([center_z_1, center_z_2]) - radius
        center_circle_y = y_1[-1]
        length_segment_2 = radius * np.cos((np.pi / 2) - np.deg2rad(curve_angle))
        y_2 = np.linspace(y_1[-1], y_1[-1] + length_segment_2, int((y_1[-1] + length_segment_2) / (0.1)) + 1,
                          endpoint=True)
        x_2 = np.ones(len(y_2)) * np.median([center_x_1, center_x_2])
        z_2 = np.sqrt(radius ** 2 - (y_2 - y_1[-1]) ** 2) + center_circle_z

        # segment 3, second straight part
        straight_lenght = 180
        m = -1 / ((z_2[-1] - center_circle_z) / (y_2[-1] - center_circle_y))
        b = (z_2[-1]) - (m * (y_2[-1]))
        length_segment_3 = straight_lenght * np.cos(m)

        y_3 = np.linspace(y_2[-1], y_2[-1] + length_segment_3, int((y_2[-1] + length_segment_3) / (0.1)) + 1,
                          endpoint=True)
        x_3 = np.ones(len(y_3)) * np.median([center_x_1, center_x_2])
        z_3 = (m * y_3) + b

        x = np.concatenate((x_1, x_2, x_3))
        y = np.concatenate((y_1, y_2, y_3))
        z = np.concatenate((z_1, z_2, z_3))

    return x, y, -z


def plot_trajectory(set_of_points, trajectory_type):

    x, y, z = build_trajectory(set_of_points, trajectory_type)
    ax = plt.axes(projection='3d')
    ax.scatter3D(x, y, z)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.figure()
    plt.subplot(121)
    plt.plot(y, z, '-o')
    plt.ylim(np.min(z)-100, np.max(z)+100)
    plt.subplot(122)
    plt.plot(y, x, '-o')


if __name__ == '__main__':
    print('notihgin')
    set_of_points = [(1, 1, 1),
                     (1, 40, 1),
                     (1, 1, 40),
                     (1, 40, 40),
                     (40, 1, 1),
                     (40, 40, 1),
                     (40, 1, 40),
                     (40, 40, 40)]

    plot_trajectory(set_of_points, 'curve_left')
    plt.show()