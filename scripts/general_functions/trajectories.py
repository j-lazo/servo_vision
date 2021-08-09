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


def straight_line(length_segment, starting_point, resolution=0.1, second_point=[], angle=0):

    """
    Builds the points of a straight line given the length, stargint point with a resolution of 0.1
    @param length_segment: (float) length of the segment in mm. The number of points int he segment is calculated as:

    num_points = (initial_point - final_point / (resolution)) + 1

    @param starting_point: (1x3 array) 3D point (x,y,z) point from where to start to draw
    @param resolution: (float), the separation between points in the segment
    @param m:
    @param angle:
    @return:
    """

    starting_point_x = starting_point[0]
    starting_point_y = starting_point[1]
    starting_point_z = starting_point[2]

    y = np.linspace(starting_point_y, starting_point_y + length_segment,
                    int((starting_point_y + length_segment) / (resolution)) + 1, endpoint=True)
    z = np.ones(len(y)) * starting_point_z

    if second_point:
        second_point_x = second_point[0]
        second_point_y = second_point[1]
        second_point_z = second_point[2]

        m = -1 / ((starting_point_z - second_point_z) / (starting_point_y - second_point_y))
        b = (starting_point_z) - (m * (starting_point_y))
        length_segment_s = length_segment * np.cos(m)

        y = np.linspace(starting_point_y, starting_point_y + length_segment_s,
                        int((starting_point_y + length_segment_s) / (resolution)) + 1, endpoint=True)
        z = (m * y) + b

    x = np.ones(len(y)) * starting_point_x
    return x, y, z


def curve(initial_point, radius, curve_angle, center_circle, resolution=0.1):

    """
    Build a curve segment given the initial point, the radius of the curve, the curve length and
    the center of the circle with resolution given i.e. separation between points selected
    @param initial_point:
    @param radius:
    @param curve_angle:
    @param center_circle:
    @return:
    """

    init_x = initial_point[0]
    init_y = initial_point[1]
    init_z = initial_point[2]

    center_circle_x = center_circle[0]
    center_circle_y = center_circle[1]
    center_circle_z = center_circle[2]

    length_segment = radius * np.cos((np.pi / 2) - np.deg2rad(curve_angle))
    y = np.linspace(init_y, init_y + length_segment, int((init_y + length_segment) / (resolution)) + 1, endpoint=True)

    x = np.ones(len(y)) * init_x
    z = np.sqrt(radius ** 2 - (y - center_circle_y) ** 2) + center_circle_z

    return x, y, z


def build_trajectory(set_of_points, trajectory_type):
    """

    @param set_of_points: set of initial 8 points
    @param trajectory_type: straight, right_curve, left_curve, s-curve
    @return: a set of spatial points (x,y,z) with the any of the previously stated trajectories
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

    if trajectory_type == 'straight_line':
        length_segment = 200
        starting_point = [np.median([center_x_1, center_x_2]),
                          starting_point_y,
                          np.median([center_z_1, center_z_2])]
        x, y, z = straight_line(length_segment, starting_point)

    if trajectory_type == 'curve_right':
        # segment 1, straight part
        length_segment_1 = 35
        starting_point_1 = [np.median([center_x_1, center_x_2]),
                          starting_point_y,
                          np.median([center_z_1, center_z_2])]

        x_1, y_1, z_1 = straight_line(length_segment_1, starting_point_1, resolution=0.1)

        # segment 2, curve
        radius = 60
        curve_angle = 15
        initial_point_2 = [np.median([center_x_1, center_x_2]),
                         y_1[-1],
                         np.median([center_z_1, center_z_2])]
        center_circle = [np.median([center_x_1, center_x_2]),
                         y_1[-1],
                         np.median([center_z_1, center_z_2]) - radius]

        x_2, y_2, z_2 = curve(initial_point_2, radius, curve_angle, center_circle)

        # segment 3, second straight part
        starting_point_3 = [x_2[-1], y_2[-1], z_2[-1]]
        x_3, y_3, z_3 = straight_line(180, starting_point_3, second_point=center_circle)

        # concatenate all the segments
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
        z = -z

    if trajectory_type == 's_curve':
        # segment 1, straight part
        length_segment_1 = 10
        y_1 = np.linspace(starting_point_y, starting_point_y + length_segment_1,
                          int((starting_point_y + length_segment_1) / (0.1)) + 1, endpoint=True)
        x_1 = np.ones(len(y_1)) * np.median([center_x_1, center_x_2])
        z_1 = np.ones(len(y_1)) * np.median([center_z_1, center_z_2])

        # segment 2, curve
        radius = 15
        curve_angle = 15
        center_circle_z = np.median([center_z_1, center_z_2]) - radius
        center_circle_y = y_1[-1]
        length_segment_2 = radius * np.cos((np.pi / 2) - np.deg2rad(curve_angle))
        y_2 = np.linspace(y_1[-1], y_1[-1] + length_segment_2, int((y_1[-1] + length_segment_2) / (0.1)) + 1,
                          endpoint=True)
        x_2 = np.ones(len(y_2)) * np.median([center_x_1, center_x_2])
        z_2 = np.sqrt(radius ** 2 - (y_2 - y_1[-1]) ** 2) + center_circle_z

        # segment 3, second straight part
        straight_lenght = 10
        m = -1 / ((z_2[-1] - center_circle_z) / (y_2[-1] - center_circle_y))
        b = (z_2[-1]) - (m * (y_2[-1]))
        length_segment_3 = straight_lenght * np.cos(m)
        y_3 = np.linspace(y_2[-1], y_2[-1] + length_segment_3, int((y_2[-1] + length_segment_3) / (0.1)) + 1,
                          endpoint=True)
        x_3 = np.ones(len(y_3)) * np.median([center_x_1, center_x_2])
        z_3 = (m * y_3) + b

        # segment 4, second curve

        radius = 15
        curve_angle = 15
        center_circle_z = z_3[-1]
        center_circle_y = y_3[-1]
        length_segment_2 = radius * np.cos((np.pi / 2) - np.deg2rad(curve_angle))
        y_4 = np.linspace(y_3[-1], y_3[-1] + length_segment_2, int((y_3[-1] + length_segment_2) /(0.1)) + 1,
                          endpoint=True)
        x_4 = np.ones(len(y_4)) * x_3[-1]
        z_4 = (np.sqrt(radius ** 2 - (y_4 - y_3[-1]) ** 2) + center_circle_z) - 30.1

        # segment 5, final liner stage

        length_segment_5 = 10
        y_5 = np.linspace(y_4[-1], y_4[-1] + length_segment_5,
                          int((y_4[-1] + length_segment_5) / (0.1)) + 1, endpoint=True)
        x_5 = np.ones(len(y_5)) * x_4[-1]
        z_5 = np.ones(len(y_5)) * z_4[-1]

        # concatenate all the segments
        x = np.concatenate((x_1, x_2, x_3, x_4, x_5))
        y = np.concatenate((y_1, y_2, y_3, y_4, y_5))
        z = np.concatenate((z_1, z_2, z_3, z_4, z_5))

    return x, y, z


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
    set_of_points = [(1, 1, 1),
                     (1, 40, 1),
                     (1, 1, 40),
                     (1, 40, 40),
                     (40, 1, 1),
                     (40, 40, 1),
                     (40, 1, 40),
                     (40, 40, 40)]

    plot_trajectory(set_of_points, 'curve_right')
    plt.show()