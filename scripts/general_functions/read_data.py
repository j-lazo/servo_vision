import pandas as pd
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
import numpy as np
from mpl_toolkits.mplot3d import axes3d


# import opencv as cv


def read_file_name():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename()

    return file_path


def read_csv(df, data_name):
    data = df[data_name].astype(int).values.tolist()

    return data

def read_jacobian_csv(df, data_name):
    data = df[data_name]

    return data


def plot2D(data_x, data_y, plot_name):
    fig, ax = plt.subplots()
    ax.plot(data_x, data_y)
    ax.plot(0, 0, 'x', color='blue')
    ax.plot(data_x[0], data_y[0], 'x', color='red')

    ax.set(xlabel='x', ylabel='y',
           title=plot_name)
    ax.set_xlim([-150, 150])
    ax.set_ylim([-150, 150])
    ax.spines['left'].set_position('zero')
    ax.spines['bottom'].set_position('zero')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.show()


def plot3D(data_x, data_y, data_z, plot_name):
    fig = plt.figure(1)
    ax = fig.add_subplot(111, aspect='equal', projection='3d')
    ax.plot3D([0], [0], [0], 'x', color='red')
    ax.plot3D([0, 0], [0, 0], [0, data_z[-1]])
    for x, y in zip(data_x, data_y):
        interval = 1
        ax.plot3D(x[::interval], y[::interval], data_z[::interval])
        print("1")
    circle_points = 50
    theta = np.linspace(0, 2 * np.pi, circle_points)
    radius = 20
    circle_x = radius * np.cos(theta)
    circle_y = radius * np.sin(theta)
    ax.plot3D(circle_x, circle_y, [0] * circle_points)
    ax.plot3D(circle_x, circle_y, [10000] * circle_points)
    ax.plot3D(circle_x, circle_y, [20000] * circle_points)
    ax.plot3D(circle_x, circle_y, [30000] * circle_points)
    ax.plot3D(circle_x, circle_y, [40000] * circle_points)
    ax.plot3D(circle_x, circle_y, [50000] * circle_points)
    ax.plot3D(circle_x, circle_y, [60000] * circle_points)
    ax.plot3D(circle_x, circle_y, [70000] * circle_points)
    ax.plot3D(circle_x, circle_y, [80000] * circle_points)
    ax.plot3D(circle_x, circle_y, [90000] * circle_points)


    ax.set(xlabel='x', ylabel='y', zlabel='z',
           title=plot_name)
    ax.set_xlim([-150, 150])
    ax.set_ylim([-150, 150])

    plt.show()


def list_substraction(array2, array1):
    array1 = np.array(array1)
    array2 = np.array(array2)

    return (array2 - array1).tolist()


def visual_to_cartesian(array1, array2, img_shape):
    w, h = img_shape
    array1 = np.array(array1)
    array2 = np.array(array2)
    array1 = array1 - w/2
    array2 = -(array2 - h/2)

    return array1.tolist(), array2.tolist()


def EM_to_cartesian(array1, array2):
    EM_x = array1
    EM_y = array2
    cartesian_x = EM_y
    cartesian_y = EM_x

    return cartesian_x, cartesian_y

def EM_to_reference_center(array1, array2, center):
    center_x, center_y = center
    amp_factor = 5
    array1 = [(x - center_x) * amp_factor for x in array1]
    array2 = [(y - center_y) * amp_factor for y in array2]

    return array1, array2


image_shape = (300, 300)

result_right_curve_good = '/home/nearlab/Jorge/current_work/robot_vision/results/n_control/straight_line/experiment_11_06_2021_18_19_straight_line_n_control/experiment_11_06_2021_18_19_.csv'
csv_file = read_file_name()
df = pd.read_csv(csv_file)
vision_point_x = read_csv(df, 'vis point x')
vision_point_y = read_csv(df, 'vis point y')
cartesian_target_point_x, cartesian_target_point_y = visual_to_cartesian(vision_point_x, vision_point_y, image_shape)
EM_sensor_x = read_csv(df, 'sensor x')
EM_sensor_y = read_csv(df, 'sensor y')
EM_sensor_z = read_csv(df, 'sensor z')
cartesian_EM_x, cartesian_EM_y = EM_to_cartesian(EM_sensor_x, EM_sensor_y)
centered_EM_x, centered_EM_y = EM_to_reference_center(cartesian_EM_x, cartesian_EM_y, (cartesian_EM_x[0], cartesian_EM_y[0]))
target_point_x = read_csv(df, 'target x')
target_point_y = read_csv(df, 'target y')
target_theta = read_csv(df, 'theta')
target_magnitude = read_csv(df, 'magnitude')
stepper_position = read_csv(df, 'stepper')
jacobian_mat = read_jacobian_csv(df, 'Jacobian')
print(type(jacobian_mat), jacobian_mat)
cut_off_range = -1

# plot2D(cartesian_target_point_x[cut_off_range:], cartesian_target_point_y[cut_off_range:], 'targets trajectory')
# plot3D(vision_point_x[cut_off_range:], vision_point_y[cut_off_range:], stepper_position[cut_off_range:], '3D Hello!')
plot3D([cartesian_target_point_x[:cut_off_range], centered_EM_x[:cut_off_range]],
       [cartesian_target_point_y[:cut_off_range], centered_EM_y[:cut_off_range]],
       stepper_position[:cut_off_range], '3D Hello!')
#
# plot3D([cartesian_target_point_x[:cut_off_range], EM_sensor_x[:cut_off_range]],
#        [cartesian_target_point_y[:cut_off_range], EM_sensor_y[:cut_off_range]],
#        stepper_position[:cut_off_range], '3D Hello!')


