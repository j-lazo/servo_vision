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


def straight_line(length_segment, starting_point, resolution=0.01, second_point=[], angle=0):

    """
    Builds the points of a straight line given the length, stargint point with a resolution of 0.1
    @param length_segment: (float) length of the segment in mm. The number of points int he segment is calculated as:

    num_points = (initial_point - final_point / (resolution)) + 1

    @param starting_point: (1x3 array) 3D point (x,y,z) point from where to start to draw
    @param resolution: (float), the separation between points in the segment
    @param second_point: if there is a second point the slope and angle of the curve will be calculated
    @return:
    """

    starting_point_x = starting_point[0]
    starting_point_y = starting_point[1]
    starting_point_z = starting_point[2]
    end_point = starting_point_y - length_segment
    number_of_points = int((np.abs(end_point - starting_point_y))/(resolution) + 1)
    y = np.around(np.linspace(starting_point_y, starting_point_y - length_segment,
                    number_of_points, endpoint=True), 2)
    z = np.ones(len(y)) * starting_point_z

    if second_point:
        second_point_x = second_point[0]
        second_point_y = second_point[1]
        second_point_z = second_point[2]

        m = -1 / ((starting_point_z - second_point_z) / (starting_point_y - second_point_y))
        b = (starting_point_z) - (m * (starting_point_y))
        end_point = starting_point_y - length_segment
        number_of_points = int((np.abs(end_point - starting_point_y)) / (resolution) + 1)
        y = np.around(np.linspace(starting_point_y, starting_point_y - length_segment,
                                  number_of_points, endpoint=True), 2)

        z = (m * y) + b

    x = np.ones(len(y)) * starting_point_x
    return x, y, z


def curve(initial_point, radius, curve_angle, center_circle, orientation='right', resolution=0.01):

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
    end_point = init_y - length_segment
    number_of_points = int((np.abs(end_point - init_y))/(resolution) + 1)
    y = np.around(np.linspace(init_y, init_y - length_segment,
                    number_of_points, endpoint=True), 2)

    x = np.ones(len(y)) * init_x
    if orientation == 'right':
        z = np.sqrt(radius ** 2 - (y - center_circle_y) ** 2) + center_circle_z

    elif orientation == 'left':
        center_circle_z = init_z + radius
        z = -(np.sqrt(radius ** 2 - (y - center_circle_y) ** 2) - center_circle_z)

    return x, y, z


def build_trajectory(set_of_points, trajectory_type, resolution=0.1):
    """

    @param set_of_points: set of initial 8 points
    @param trajectory_type: straight, right_curve, left_curve, s-curve
    @return: a set of spatial points (x,y,z) with the any of the previously stated trajectories
    """
    # lower square
    poitn_0 = set_of_points[0]
    poitn_1 = set_of_points[1]
    poitn_2 = set_of_points[2]
    poitn_3 = set_of_points[3]

    # upper square
    poitn_4 = set_of_points[4]
    poitn_5 = set_of_points[5]
    poitn_6 = set_of_points[6]
    poitn_7 = set_of_points[7]

    center_y_1 = get_centers(poitn_1, poitn_2, 'y')
    center_y_2 = get_centers(poitn_3, poitn_4, 'y')

    center_z_1 = get_centers(poitn_1, poitn_3, 'z')
    center_z_2 = get_centers(poitn_2, poitn_4, 'z')

    center_x_1 = get_centers(poitn_1, poitn_3, 'x')
    center_x_2 = get_centers(poitn_2, poitn_4, 'x')

    starting_point_x = np.mean([(poitn_0[0] + poitn_4[0])/2, (poitn_2[0] + poitn_6[0])/2])
    starting_point_y = np.mean([poitn_0[1], poitn_2[1], poitn_4[1], poitn_6[1]]) + 10
    starting_point_z = np.mean([(poitn_0[2] + poitn_2[2])/2, (poitn_3[2] + poitn_4[2])/2])

    if trajectory_type == 'straight_line':
        length_segment = 120
        starting_point = [round(starting_point_x, 2),
                          round(starting_point_y, 2),
                          round(starting_point_z, 2)]
        x, y, z = straight_line(length_segment, starting_point, resolution=0.01)

    if trajectory_type == 'curve_right':
        # segment 1, straight part
        length_segment_1 = 35
        starting_point_1 = [starting_point_x,
                            starting_point_y,
                            starting_point_z]

        x_1, y_1, z_1 = straight_line(length_segment_1, starting_point_1, resolution=0.01)

        # segment 2, curve
        radius = 60
        curve_angle = 20
        initial_point_2 = [starting_point_x,
                           y_1[-1],
                           starting_point_z]
        center_circle = [starting_point_x,
                         y_1[-1],
                         starting_point_z - radius]

        x_2, y_2, z_2 = curve(initial_point_2, radius, curve_angle, center_circle, resolution=0.01)

        # segment 3, second straight part
        starting_point_3 = [x_2[-1], y_2[-1], z_2[-1]]
        length_segment_3 = 65
        x_3, y_3, z_3 = straight_line(length_segment_3, starting_point_3, resolution=0.01,
                                      second_point=center_circle)

        # concatenate all the segments
        x = np.concatenate((x_1, x_2, x_3))
        y = np.concatenate((y_1, y_2, y_3))
        z = np.concatenate((z_1, z_2, z_3))

    if trajectory_type == 'curve_left':
        # segment 1, straight part
        length_segment_1 = 35
        starting_point_1 = [starting_point_x,
                            starting_point_y,
                            starting_point_z]

        x_1, y_1, z_1 = straight_line(length_segment_1, starting_point_1, resolution=0.01)

        # segment 2, curve
        radius = 60
        curve_angle = 15
        initial_point_2 = [starting_point_x,
                           y_1[-1],
                           starting_point_z]
        center_circle = [starting_point_x,
                         y_1[-1],
                         starting_point_z + radius]

        x_2, y_2, z_2 = curve(initial_point_2, radius, curve_angle, center_circle, orientation='left', resolution=0.01)

        # segment 3, second straight part

        starting_point_3 = [x_2[-1], y_2[-1], z_2[-1]]
        length_segment_3 = 70
        x_3, y_3, z_3 = straight_line(length_segment_3, starting_point_3, resolution=0.01,
                                      second_point=center_circle)

        x = np.concatenate((x_1, x_2, x_3))
        y = np.concatenate((y_1, y_2, y_3))
        z = np.concatenate((z_1, z_2, z_3))
        z = z

    if trajectory_type == 's_curve':
        # segment 1, straight part
        length_segment_1 = 15
        starting_point_1 = [starting_point_x,
                            starting_point_y,
                            starting_point_z - 6]

        x_1, y_1, z_1 = straight_line(length_segment_1, starting_point_1, resolution=0.01)

        # segment 2, curve
        radius = 60
        curve_angle = 30
        initial_point_2 = [starting_point_x,
                           y_1[-1],
                           starting_point_z - 6]
        center_circle = [starting_point_x,
                         y_1[-1],
                         starting_point_z + radius]

        x_2, y_2, z_2 = curve(initial_point_2, radius, curve_angle, center_circle,
                              orientation='left', resolution=0.01)

        # segment 3, second straight part
        starting_point_3 = [x_2[-1], y_2[-1], z_2[-1]]
        length_segment_3 = 2
        x_3, y_3, z_3 = straight_line(length_segment_3, starting_point_3, resolution=0.01,
                                      second_point=center_circle)

        # concatenate all the segments
        x = np.concatenate((x_1, x_2, x_3))
        y = np.concatenate((y_1, y_2, y_3))
        z = np.concatenate((z_1, z_2, z_3))
        # segment 4, second curve

        radius_2 = 60
        curve_angle_2 = 30
        initial_point_2 = [starting_point_x,
                           y_3[-1],
                           starting_point_z]
        center_circle = [starting_point_x,
                         y_3[-1]-35,
                         starting_point_z
                         - radius_2 + 8.5]

        x_4, y_4, z_4 = curve(initial_point_2, radius_2, curve_angle_2, center_circle,
                              orientation='right', resolution=0.005)

        # segment 5, final linear stage
        length_segment_5 = 50
        starting_point_5 = [x_4[-1], y_4[-1], z_4[-1]]
        x_5, y_5, z_5 = straight_line(length_segment_5, starting_point_5, resolution=0.01)

        # concatenate all the segments
        x = np.concatenate((x_1, x_2, x_3, x_4, x_5))
        y = np.round(np.concatenate((y_1, y_2, y_3, y_4, y_5)), 2)
        z = np.concatenate((z_1, z_2, z_3, z_4, z_5))

    return x, y, z


def plot_trajectory(set_of_points, trajectory_type):

    """
    Function just to show the trajectory, it calls build trajectory and plot the generated points
    @param set_of_points: set of initial 8 points of a cube
    @param trajectory_type: straight, right_curve, left_curve, s-curve
    @return: plot with the
    """

    x, y, z = build_trajectory(set_of_points, trajectory_type)

    fig = plt.figure()
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224, projection='3d')

    ax1.plot(y, z, '-o')
    ax1.set_ylim(np.min(z)-100, np.max(z)+100)
    ax1.title.set_text('Y vs Z')
    ax1.set_xlabel('y')
    ax1.set_ylabel('z')

    ax2.plot(y, x, '-o')
    ax2.title.set_text('Y vs X')
    ax2.set_xlabel('y')
    ax2.set_ylabel('x')

    ax3.plot(z, x, '-o')
    #ax3.title.set_text('Z vs X')
    ax3.set_xlabel('z')
    ax3.set_ylabel('x')

    ax4.scatter3D(y, z, x)
    ax4.title.set_text('3D View')
    ax4.set_xlabel('Y')
    ax4.set_ylabel('Z')
    ax4.set_zlabel('X')


if __name__ == '__main__':
    # sample to show how the trajectory generator considering an initial cube of 40x40x40 mm
    set_of_points = [(1, 1, 1),
                     (1, -40, 1),
                     (1, 1, -40),
                     (1, -40, -40),
                     (-40, 1, 1),
                     (-40, -40, 1),
                     (-40, 1, -40),
                     (-40, -40, -40)]

    plot_trajectory(set_of_points, 'curve_right')
    plt.show()
    #test_best_fit()
    #test_icp()
