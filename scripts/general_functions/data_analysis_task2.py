import os
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import glob
import time
import data_managament as dm
import scipy.stats as stats
from scipy.stats import kruskal
from datetime import datetime
from mpl_toolkits import mplot3d


class GroundTruthSPath:
    def __init__(self, x, x_shift=0, y_shift=0, length1=0, length2=20, radius=60, angle=20, origin=0):
        self.length1 = length1
        self.length2 = length2
        self.radius = radius
        self.angle = angle * 2 * np.pi / 360
        self.x = x - x_shift
        self.x_shift = x_shift
        self.y_shift = y_shift

        if x[-1] > x[0]:
            self.__index1 = (next(i for i, v in enumerate(x) if v > x[0] + length1))
            self.__index2 = (
                next(i for i, v in enumerate(x) if v > (x[0] + length1 + radius * np.sin(self.angle))))
            self.__index3 = (
                next(i for i, v in enumerate(x) if v > (x[0] + length1 +
                                                        radius * np.sin(self.angle) +
                                                        length2 * np.cos(self.angle))))
            self.__index4 = (
                next(i for i, v in enumerate(x) if v > (x[0] + length1 +
                                                        radius * np.sin(self.angle) +
                                                        length2 * np.cos(self.angle) +
                                                        radius * np.sin(self.angle))))
        elif x[-1] < x[0]:
            self.__index1 = (next(i for i, v in enumerate(x) if v < x[0] - length1))
            self.__index2 = (
                next(i for i, v in enumerate(x) if v < (x[0] - length1 - radius * np.sin(self.angle))))
            self.__index3 = (
                next(i for i, v in enumerate(x) if v < (x[0] - length1 -
                                                        radius * np.sin(self.angle) -
                                                        length2 * np.cos(self.angle))))
            self.__index4 = (
                next(i for i, v in enumerate(x) if v < (x[0] - length1 -
                                                        radius * np.sin(self.angle) -
                                                        length2 * np.cos(self.angle) -
                                                        radius * np.sin(self.angle))))

        self.x1 = np.array([])
        self.x2 = np.array([])
        self.x3 = np.array([])
        self.x4 = np.array([])
        self.x5 = np.array([])
        self.y = np.array([])
        self.y1 = np.array([])
        self.y2 = np.array([])
        self.y3 = np.array([])
        self.y4 = np.array([])
        self.y5 = np.array([])

    def plot(self):
        plt.plot(self.x1, self.y1, 'r')
        plt.plot(self.x2, self.y2, 'g')
        plt.plot(self.x3, self.y3, 'b')
        plt.plot(self.x4, self.y4, 'r')
        plt.plot(self.x5, self.y5, 'g')

    def x_init(self):
        self.x1 = self.x[0:self.__index1]
        self.x2 = self.x[self.__index1:self.__index2]
        self.x3 = self.x[self.__index2:self.__index3]
        self.x4 = self.x[self.__index3:self.__index4]
        self.x5 = self.x[self.__index4:]

    def y_init(self, x_flip=True, y_flip=False):
        if not x_flip and not y_flip:
            self.y1 = self.y_shift * np.ones(len(self.x1))  # doesn't not change with x shift
            self.y2 = np.sqrt(self.radius ** 2 - (self.x2 - self.x1[-1]) ** 2) + (-self.radius + self.y_shift)
            self.y3 = -np.tan(self.angle) * (self.x3 - self.x2[-1]) + self.y2[-1]
            self.y4 = np.sqrt(self.radius ** 2 - (self.x4 - self.x3[-1] - np.sin(self.angle)) ** 2) + (
                        self.radius * np.cos(self.angle) + self.y_shift)
            self.y5 = self.y_shift * np.ones(len(self.x5))

        elif x_flip and not y_flip:
            self.y1 = self.y_shift * np.ones(len(self.x1))  # doesn't not change with x shift
            self.y2 = -(np.sqrt(self.radius ** 2 - (self.x2 - self.x1[-1]) ** 2) + (-self.radius + self.y_shift))
            self.y3 = -np.tan(self.angle) * (self.x3 - self.x2[-1]) + self.y2[-1]
            self.y4 = np.sqrt(self.radius ** 2 - (self.x4 - self.x3[-1] + self.radius * np.sin(self.angle)) ** 2) + (
                    -self.radius * np.cos(self.angle) + self.y3[-1] + self.y_shift)
            self.y5 = self.y4[-1] * np.ones(len(self.x5))

        self.y = np.concatenate((self.y1, self.y2, self.y3, self.y4, self.y5), axis=None)

    def shift_x(self, dx):
        self.x = self.x - dx
        self.x1 = self.x[0:self.__index1]
        self.x2 = self.x[self.__index1:self.__index2]
        self.x3 = self.x[self.__index2:]
        return

    def shift_y(self, dy):
        self.y = self.y + dy
        return

    def diff(self, y):
        return np.abs(np.subtract(y, self.y)).sum() / len(y)


class GroundTruth:
    def __init__(self, x, x_shift=0, y_shift=0, length=40, radius=60, angle=15, origin=0):
        if x[-1] > x[0]:
            self.__index1 = (next(i for i, v in enumerate(x) if v > x[0] + length))
            self.__index2 = (
                next(i for i, v in enumerate(x) if v > (x[0] + length + radius * np.sin(2 * np.pi / 360 * angle))))
        elif x[-1] < x[0]:
            self.__index1 = (next(i for i, v in enumerate(x) if v < x[0] - length))
            self.__index2 = (
                next(i for i, v in enumerate(x) if v < (x[0] - length - radius * np.sin(2 * np.pi / 360 * angle))))
        self.length = length
        self.radius = radius
        self.angle = angle
        self.x = x - x_shift
        self.x_shift = x_shift
        self.y_shift = y_shift
        self.x1 = np.array([])
        self.x2 = np.array([])
        self.x3 = np.array([])
        self.y = np.array([])
        self.y1 = np.array([])
        self.y2 = np.array([])
        self.y3 = np.array([])

    def plot(self):
        # plt.figure(figsize=(24, 2))
        plt.plot(self.x1, self.y1, 'r')
        plt.plot(self.x2, self.y2, 'g')
        plt.plot(self.x3, self.y3, 'b')

    def x_init(self):
        self.x1 = self.x[0:self.__index1]
        self.x2 = self.x[self.__index1:self.__index2]
        self.x3 = self.x[self.__index2:]

    def y_init(self, y_flip=False, x_flip=False):
        if not x_flip and not y_flip:
            self.y1 = self.y_shift * np.ones(len(self.x1))  # doesn't not change with x shift
            self.y2 = np.sqrt(self.radius ** 2 - (self.x2 - self.x1[-1]) ** 2) + (-self.radius + self.y_shift)
            self.y3 = -np.tan(2 * np.pi / 360 * 15) * (self.x3 - self.x2[-1]) + self.y2[-1]
        elif x_flip and not y_flip:
            self.y1 = self.y_shift * np.ones(len(self.x1))  # doesn't not change with x shift
            self.y2 = np.sqrt(self.radius ** 2 - (self.x2 - self.x1[-1]) ** 2) + (-self.radius + self.y_shift)
            self.y3 = np.tan(2 * np.pi / 360 * 15) * (self.x3 - self.x2[-1]) + self.y2[-1]

        elif not x_flip and y_flip:
            self.y1 = -(self.y_shift * np.ones(len(self.x1)))  # doesn't not change with x shift
            self.y2 = -(np.sqrt(self.radius ** 2 - (self.x2 - self.x1[-1]) ** 2) + (-self.radius + self.y_shift))
            self.y3 = -(-np.tan(2 * np.pi / 360 * 15) * (self.x3 - self.x2[-1]) + self.y2[-1])

        elif x_flip and y_flip:
            self.y1 = -(self.y_shift * np.ones(len(self.x1)))  # doesn't not change with x shift
            self.y2 = -(np.sqrt(self.radius ** 2 - (self.x2 - self.x1[-1]) ** 2) + (-self.radius + self.y_shift))
            self.y3 = -(np.tan(2 * np.pi / 360 * 15) * (self.x3 - self.x2[-1]) - self.y2[-1])

        self.y = np.concatenate((self.y1, self.y2, self.y3), axis=None)

    def shift_x(self, dx):
        self.x = self.x - dx
        self.x1 = self.x[0:self.__index1]
        self.x2 = self.x[self.__index1:self.__index2]
        self.x3 = self.x[self.__index2:]
        return

    def shift_y(self, dy):
        self.y = self.y + dy
        return

    def diff(self, y):
        return np.abs(np.subtract(y, self.y)).sum() / len(y)

    def diff_area(self, y):
        sum_area = 0.0
        for i in range(len(self.x) - 1):
            dx = self.x[i+1] - self.x[i]
            dy0 = abs(y[i] - self.y[i])
            dy1 = abs(y[i+1] - self.y[i+1])
            area = (dy0 + dy1) * dx / 2
            sum_area = sum_area + area
        return sum_area

def calculate_kruskal_p_value(parameter_data_1, parameter_data_2):
    result = stats.ttest_ind(a=parameter_data_1,
                             b=parameter_data_2,
                             equal_var=False)
    print('T-test result')
    print(result)

    # compare samples
    stat, p = kruskal(parameter_data_1,
                      parameter_data_2)
    print('Statistics=%.9f, p=%.9f' % (stat, p))
    # interpret
    print('otra vez')
    print(stat, p)
    alpha = 0.05
    if p > alpha:
        print('Same distributions (fail to reject H0)')
    else:
        print('Different distributions (reject H0)')


def analyze_data_single_experiment(file_directory):
    json_file_dir = file_directory + 'calibration_data.json'
    files_in_dir = os.listdir(file_directory)
    json_file = [file for file in files_in_dir if file.endswith('.json')][0]
    csv_file = [file for file in files_in_dir if file.endswith('.csv')][0]
    calibration_data = dm.read_data_json(file_directory + json_file)

    experiment_data = pd.read_csv(file_directory + csv_file, skipinitialspace=True)

    sensor_1_x = experiment_data['sensor 1 x'].tolist()
    sensor_2_x = experiment_data['sensor 2 x'].tolist()
    sensor_3_x = experiment_data['sensor 3 x'].tolist()

    sensor_1_y = experiment_data['sensor 1 y'].tolist()
    sensor_2_y = experiment_data['sensor 2 y'].tolist()
    sensor_3_y = experiment_data['sensor 3 y'].tolist()

    sensor_1_z = experiment_data['sensor 1 z'].tolist()
    sensor_2_z = experiment_data['sensor 2 z'].tolist()
    sensor_3_z = experiment_data['sensor 3 z'].tolist()

    plt.figure()
    plt.plot(calibration_data['limit_point_z'][:24], calibration_data['limit_point_y'][:24], 'o')
    plt.plot(sensor_1_z, sensor_1_y, 'r-*')
    plt.plot(sensor_2_z, sensor_2_y, 'y-*')
    plt.plot(sensor_3_z, sensor_3_y, 'g-*')


def discretise_series_horizontal(data_list):
    min_value = np.amin([np.amin(sub_list) for sub_list in data_list])
    max_value = np.amax([np.amax(sub_list) for sub_list in data_list])
    num_points = (int(max_value) + 1 - (int(min_value) - 1)) * 2
    discretised_series = np.linspace(int(min_value) - 1, int(max_value) + 1, num_points, endpoint=False)
    return list(discretised_series)


def discretise_series_vertical(discreet_x, list_points_x, list_points_y):
    print(np.shape(discreet_x))
    discreet_y = [[] for _ in range(len(discreet_x))]
    for index_list, originals_horizontal in enumerate(list_points_x):
        for index_point, point in enumerate(originals_horizontal):
            for point_l in discreet_x:
                if point_l <= point < point_l + 0.5:
                    index_discreet = (discreet_x.index(point_l))
                    discreet_y[index_discreet].append(list_points_y[index_list][index_point])

    for j, sub_list in enumerate(discreet_y):
        if sub_list == []:
            discreet_y[j] = discreet_y[j - 1]
    mean_y = [np.mean(sub_list) for sub_list in discreet_y]
    stdv_y = [np.std(sub_list) for sub_list in discreet_y]

    return discreet_y, mean_y, stdv_y


def discretise_single_series_vertical(teoretical_series, list_points_series):
    print(teoretical_series)
    print(list_points_series)
    discreet_series = [[] for _ in range(len(teoretical_series))]
    error_sries = [[] for _ in range(len(teoretical_series))]

    for index_point, point in enumerate(list_points_series):
        for point_l in teoretical_series:
            if point_l <= point < point_l + 0.5:
                index_discreet = (teoretical_series.index(point_l))
                discreet_series[index_discreet].append(list_points_series[index_point])

    for j, sub_list in enumerate(discreet_series):
        if sub_list == []:
            print('True')
            discreet_series[j] = discreet_series[j - 1]

    return discreet_series, error_sries


def build_ground_truth_paths(directory_files, plot=True):
    list_files = os.listdir(directory_files)
    list_files = sorted([file for file in list_files if file.endswith('.csv')])
    list_all_x = []
    list_all_y = []
    list_all_z = []
    for file in list_files:
        data = pd.read_csv(directory_files + file, skipinitialspace=True)
        list_all_x.append(data['sensor x'].tolist())
        list_all_y.append(data['sensor y'].tolist())
        list_all_z.append(data['sensor z'].tolist())

    min_y = np.amin([np.amin(sub_list) for sub_list in list_all_z])
    max_y = np.amax([np.amax(sub_list) for sub_list in list_all_z])
    min_x = np.amin([np.amin(sub_list) for sub_list in list_all_y])
    max_x = np.amax([np.amax(sub_list) for sub_list in list_all_y])

    inner_radius = 5
    outer_radius = 10

    line_x = list(range(int(min_x) - 2, int(max_x) + 2))
    line_y = np.ones((len(line_x))) * (min_y + (max_y - min_y) * 0.5)
    upper_lim_1 = [line + inner_radius for line in line_y]
    upper_lim_2 = [line + outer_radius for line in line_y]
    down_lim_1 = [line - inner_radius for line in line_y]
    down_lim_2 = [line - outer_radius for line in line_y]

    discrete_series_x = discretise_series_horizontal(list_all_y)
    discreet_y, mean_y, stdv_y = discretise_series_vertical(discrete_series_x, list_all_y, list_all_z)
    up_error = [mean_value + stdv_y[j] for j, mean_value in enumerate(mean_y)]
    down_error = [mean_value - stdv_y[j] for j, mean_value in enumerate(mean_y)]

    if plot is True:
        plt.figure()
        # for j in range(len(list_all_z)):
        #    plt.plot(list_all_y[j], list_all_z[j], 'o')
        plt.plot(discrete_series_x, mean_y, '--')
        plt.plot(line_x, down_lim_1, color='darkorange')
        plt.plot(line_x, upper_lim_1, color='darkorange')
        plt.plot(line_x, upper_lim_2, color='darkorange')
        plt.plot(line_x, down_lim_2, color='darkorange')
        plt.fill_between(line_x, upper_lim_1, down_lim_1, alpha=0.4, color='bisque')
        plt.fill_between(line_x, upper_lim_2, upper_lim_1, alpha=0.8, color='darkorange')
        plt.fill_between(line_x, down_lim_1, down_lim_2, alpha=0.8, color='darkorange')
        plt.fill_between(discrete_series_x, up_error, down_error, alpha=0.3, color='blue')

        plt.ylim([min_y - 30, max_y + 30])
        # plt.axis('off')
        """fig1, axs1 = plt.subplots(len(list_files), 1, figsize=(17, 8))
        for j, model in enumerate(list_files):
            for i in range(len(list_all_z)):
                print(i, j)
                print(list_all_z[i])
                axs1[j].plot(list_all_z[j])
                #axs1[i, j].set_ylim([0, 1.1])
                #axs1[i, j].set_title(''.join(
                #    ['bs: ', model['batch size'], '\n', 'lr: ', "{:.1e}".format(float(model['learning rate']))]))
                #if j != 0:
                #    axs1[i, j].set_yticks([])
                #axs1[i, j].set_xticks([])"""


def plot_3D_trajectory(directory_results, directory_file):
    list_files = os.listdir(directory_results)
    list_files = sorted([file for file in list_files if file.endswith('.csv')])
    list_all_x = []
    list_all_y = []
    list_all_z = []
    for file in list_files:
        data = pd.read_csv(directory_results + file, skipinitialspace=True)
        list_all_x.append(data['sensor x'].tolist())
        list_all_y.append(data['sensor y'].tolist())
        list_all_z.append(data['sensor z'].tolist())

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    min_y = np.amin([np.amin(sub_list) for sub_list in list_all_z])
    max_y = np.amax([np.amax(sub_list) for sub_list in list_all_z])
    min_x = np.amin([np.amin(sub_list) for sub_list in list_all_y])
    max_x = np.amax([np.amax(sub_list) for sub_list in list_all_y])

    inner_radius = 8
    outer_radius = 12

    discrete_series_x = discretise_series_horizontal(list_all_y)
    discreet_y, mean_y, stdv_y = discretise_series_vertical(discrete_series_x, list_all_y, list_all_z)
    discreet_x, mean_x, stdv_x = discretise_series_vertical(discrete_series_x, list_all_y, list_all_x)

    # Cylinder
    x_cilinder = np.linspace(68, 84, 500)
    z = np.linspace(min_x - 7, max_x + 7, 500)
    Xc, Zc = np.meshgrid(x_cilinder, z)
    Yc1 = np.sqrt(inner_radius ** 2 - (76 - Xc) ** 2) - 158
    Yc2 = -np.sqrt(inner_radius ** 2 - (76 - Xc) ** 2) - 158

    out_x_cilinder = np.linspace(64, 88, 500)
    z = np.linspace(min_x - 7, max_x + 7, 500)
    Xc2, Zc2 = np.meshgrid(out_x_cilinder, z)
    Yc3 = np.sqrt(outer_radius ** 2 - (76 - Xc2) ** 2) - 158
    Yc4 = -np.sqrt(outer_radius ** 2 - (76 - Xc2) ** 2) - 158

    x_line = np.ones(500) * 76
    z_line = np.ones(500) * -158
    y_line = np.linspace(-15, 140, 500)

    # Draw parameters
    rstride = 10
    cstride = 20
    # plot the cylinders
    ax.plot_wireframe(Xc, Zc, Yc1, alpha=0.2, rstride=rstride, cstride=cstride, color='orange')
    ax.plot_wireframe(Xc, Zc, Yc2, alpha=0.2, rstride=rstride, cstride=cstride, color='orange')
    ax.plot_wireframe(Xc2, Zc2, Yc3, alpha=0.1, rstride=rstride, cstride=cstride, color='orange')
    ax.plot_wireframe(Xc2, Zc2, Yc4, alpha=0.1, rstride=rstride, cstride=cstride, color='orange')

    ax.plot(x_line, y_line, z_line, 'r-o')
    for j in range(len(list_all_x)):
        ax.plot(list_all_x[j], list_all_y[j], list_all_z[j], '-o', label='trajectory ' + str(j))
    ax.legend()
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_xlim(55, 95)
    plt.show()


def calcualte_error_list(ground_truth, measured):
    ground_truth = list(ground_truth)
    discreet_list = [[] for _ in range(len(ground_truth))]

    for index_point, point in enumerate(measured):
        for point_l in ground_truth:
            if point_l <= point < point_l + 0.5:
                index_discreet = (ground_truth.index(point_l))
                discreet_list[index_discreet].append(measured[index_point])

    for j, sub_list in enumerate(discreet_list):
        if sub_list == []:
            discreet_list[j] = discreet_list[j - 1]

    mean_y = [np.mean(sub_list) for sub_list in discreet_list]
    abs_error_list = [(np.abs(value - ground_truth[j])) for j, value in enumerate(mean_y)]

    return abs_error_list


def calcualte_error_dependant_list(ground_truth, measured, auxiliar_list):
    ground_truth = list(ground_truth)
    discreet_list = [[] for _ in range(len(ground_truth))]

    for index_point, point in enumerate(measured):
        for point_l in ground_truth:
            # print(point_l, point)
            if point_l <= point < point_l + 5:
                index_discreet = (ground_truth.index(point_l))
                discreet_list[index_discreet].append(measured[index_point])

    for j, sub_list in enumerate(discreet_list):
        if sub_list == []:
            discreet_list[j] = discreet_list[j - 1]

    mean_y = [np.mean(sub_list) for sub_list in discreet_list]
    abs_error_list = [(np.abs(value - ground_truth[j])) for j, value in enumerate(mean_y)]
    return abs_error_list


def compare_gt_and_results(directory_files, results_directory, plot=True):
    list_files = os.listdir(directory_files)
    list_files = sorted([file for file in list_files if file.endswith('.csv')])
    list_all_x = []
    list_all_y = []
    list_all_z = []
    for file in list_files:
        data = pd.read_csv(directory_files + file, skipinitialspace=True)
        list_all_x.append(data['sensor x'].tolist())
        list_all_y.append(data['sensor y'].tolist())
        list_all_z.append(data['sensor z'].tolist())

    min_y = np.amin([np.amin(sub_list) for sub_list in list_all_z])
    max_y = np.amax([np.amax(sub_list) for sub_list in list_all_z])
    min_x = np.amin([np.amin(sub_list) for sub_list in list_all_y])
    max_x = np.amax([np.amax(sub_list) for sub_list in list_all_y])

    min_z = np.amin([np.amin(sub_list) for sub_list in list_all_x])
    max_z = np.amax([np.amax(sub_list) for sub_list in list_all_x])

    inner_radius = 8
    outer_radius = 14

    discrete_series_x = discretise_series_horizontal(list_all_y)
    discreet_y, mean_y, stdv_y = discretise_series_vertical(discrete_series_x, list_all_y, list_all_z)
    discreet_z, mean_z, stdv_z = discretise_series_vertical(discrete_series_x, list_all_y, list_all_x)

    up_error = [mean_value + stdv_y[j] * 2 for j, mean_value in enumerate(mean_y)]
    down_error = [mean_value - stdv_y[j] * 2 for j, mean_value in enumerate(mean_y)]

    line_x = np.linspace(int(min_x) - 2, int(max_x) + 2, len(mean_y))
    line_y = np.ones((len(line_x))) * (min_y + (max_y - min_y) * 0.5)
    line_z = np.ones((len(line_x))) * (min_z + (max_z - min_z) * 0.5)

    print(np.shape(line_x))
    print(np.shape(line_y))

    upper_lim_1 = [line + inner_radius for line in mean_y]
    upper_lim_2 = [line + outer_radius for line in mean_y]
    down_lim_1 = [line - inner_radius for line in mean_y]
    down_lim_2 = [line - outer_radius for line in mean_y]

    x_curve = np.linspace(discrete_series_x[-1], discrete_series_x[-1] + inner_radius, 180)
    x_curve_2 = np.linspace(discrete_series_x[-1], discrete_series_x[-1] + outer_radius, 180)

    y_curve = [np.sqrt(inner_radius ** 2 - (x - discrete_series_x[-1]) ** 2) + mean_y[-1] for x in x_curve]
    y_curve_2 = [-np.sqrt(inner_radius ** 2 - (x - discrete_series_x[-1]) ** 2) + mean_y[-1] for x in x_curve]

    y_curve_3 = [np.sqrt(outer_radius ** 2 - (x - discrete_series_x[-1]) ** 2) + mean_y[-1] for x in x_curve_2]
    y_curve_4 = [-np.sqrt(outer_radius ** 2 - (x - line_x[-1]) ** 2) + mean_y[-1] for x in x_curve_2]

    results_csv = []
    list_sub_folders = sorted(os.listdir(results_directory))
    for sub_folder in list_sub_folders:
        results_csv.append(glob.glob(os.path.join(results_directory, sub_folder) + '/*.csv')[0])

    list_files = os.listdir(results_directory)
    list_files = [file for file in list_files if file.endswith('.csv')]
    list_all_x = []
    list_all_y = []
    list_all_z = []
    for csv_file in results_csv:
        data = pd.read_csv(csv_file, skipinitialspace=True)
        list_all_x.append(data['sensor x'].tolist())
        list_all_y.append(data['sensor y'].tolist())
        list_all_z.append(data['sensor z'].tolist())

    if plot is True:
        plt.figure()
        plt.plot(discrete_series_x[6:], mean_y[6:], '--')
        plt.fill_between(discrete_series_x, up_error, down_error, alpha=0.3, color='blue')

        # plt.plot(line_x, line_y, 'r--')
        plt.plot(discrete_series_x, down_lim_1, color='darkorange', linewidth=3)
        plt.plot(discrete_series_x, upper_lim_1, color='darkorange', linewidth=3)
        plt.plot(discrete_series_x, upper_lim_2, color='darkorange', linewidth=3)
        plt.plot(discrete_series_x, down_lim_2, color='darkorange', linewidth=3)

        plt.plot(x_curve, y_curve, color='darkorange')
        plt.plot(x_curve, y_curve_2, color='darkorange')
        plt.plot(x_curve_2, y_curve_3, color='darkorange')
        plt.plot(x_curve_2, y_curve_4, color='darkorange')
        plt.fill_between(x_curve_2, y_curve, y_curve_3, alpha=0.8, color='darkorange')
        plt.fill_between(x_curve_2, y_curve_3, y_curve_4, alpha=0.2, color='orange')
        plt.fill_between(x_curve, y_curve, y_curve_2, alpha=0.9, color='bisque')

        plt.fill_between(discrete_series_x, upper_lim_1, down_lim_1, alpha=0.4, color='bisque')
        plt.fill_between(discrete_series_x, upper_lim_2, upper_lim_1, alpha=0.8, color='darkorange')
        plt.fill_between(discrete_series_x, down_lim_1, down_lim_2, alpha=0.8, color='darkorange')

        for j in range(len(list_all_z)):
            plt.plot(list_all_y[j], list_all_z[j], 'o-')

        plt.ylim([min_y - 30, max_y + 30])
        plt.xlabel('position y')
        plt.ylabel('position x')

        plt.figure()
        # plt.title('Error Y axis')
        for j in range(len(list_all_z)):
            line_temp = list_all_y[j][::-1]
            plot_list = calcualte_error_list(discrete_series_x, line_temp)
            plt.plot(plot_list, 'o-', label='trajectory ' + str(j))
            plt.ylabel('Error Y (mm)')
            plt.legend(loc='best')
            plt.xlabel('t')

        plt.figure()
        # plt.title('Error Y axis')
        for j in range(len(list_all_z)):
            line_temp = list_all_x[j]  # [::-1]
            # plot_list = [np.abs(x - line_z[0]) for x in line_temp]
            plot_list = calcualte_error_dependant_list(mean_z, line_temp, discrete_series_x)
            plt.plot(discrete_series_x, plot_list, 'o-', label='trajectory ' + str(j))
            plt.ylabel('Absolute error X (mm)')
            plt.legend(loc='best')
            plt.xlim(max(discrete_series_x), min(discrete_series_x))
            plt.xlabel('y (mm)')

        plt.figure()
        # plt.title('Error Y axis')
        for j in range(len(list_all_z)):
            line_temp = list_all_z[j]  # [::-1]
            # plot_list = [np.abs(x - line_y[0]) for x in line_temp]
            plot_list = calcualte_error_dependant_list(mean_y, line_temp, discrete_series_x)
            plt.plot(discrete_series_x, plot_list, 'o-', label='trajectory ' + str(j))
            plt.ylabel('Absolute error Z (mm)')
            plt.legend(loc='best')
            plt.gca().invert_xaxis()
            plt.xlabel('y (mm)')


def analyze_results(directory_files, plot=True):
    results_csv = []
    list_sub_folders = sorted(os.listdir(directory_files))
    for sub_folder in list_sub_folders:
        results_csv.append(glob.glob(os.path.join(directory_files, sub_folder) + '/*.csv')[0])

    print(results_csv)
    # list_results_files = sorted([file for file in list_results_files if file.endswith('.csv')])
    # results_csv = sorted([file for file in results_csv if file.endswith('.csv')])
    list_files = os.listdir(directory_files)
    list_files = [file for file in list_files if file.endswith('.csv')]
    list_all_x = []
    list_all_y = []
    list_all_z = []
    for csv_file in results_csv:
        data = pd.read_csv(csv_file, skipinitialspace=True)
        list_all_x.append(data['sensor x'].tolist())
        list_all_y.append(data['sensor y'].tolist())
        list_all_z.append(data['sensor z'].tolist())

    min_y, max_y = np.amin(np.amin(list_all_z)), np.amax(np.amax(list_all_z))
    if plot is True:
        plt.figure()
        for j in range(len(list_all_z)):
            plt.subplot(2, 1, 1)
            plt.plot(list_all_z[j], 'o-')
            # plt.plot(list_all_y[j], list_all_z[j], 'o-')
            # plt.ylim([-250, 150])
            plt.subplot(2, 1, 2)
            plt.plot(list_all_x[j], 'o-')

        """fig1, axs1 = plt.subplots(len(list_files), 1, figsize=(17, 8))
        for j, model in enumerate(list_files):
            for i in range(len(list_all_z)):
                print(i, j)
                print(list_all_z[i])
                axs1[j].plot(list_all_z[j])
                #axs1[i, j].set_ylim([0, 1.1])
                #axs1[i, j].set_title(''.join(
                #    ['bs: ', model['batch size'], '\n', 'lr: ', "{:.1e}".format(float(model['learning rate']))]))
                #if j != 0:
                #    axs1[i, j].set_yticks([])
                #axs1[i, j].set_xticks([])"""


def calculate_centroid_triangle(edge_1_y, edge_1_z, edge_1_x,
                                edge_2_y, edge_2_x,
                                edge_3_y, edge_3_x):
    ox = []
    oy = []
    oz = edge_1_z[:133]
    for j, point in enumerate(oz):
        oy.append((edge_1_y[j] + edge_2_y[j] + edge_3_y[j]) / 3)
        ox.append((edge_1_x[j] + edge_2_x[j] + edge_3_x[j]) / 3)

    return ox, oy, oz


def calculate_average(list_points):
    data = np.array(list_points)
    return np.average(data, axis=0)


def plot_3D_data(directory_results):
    list_files = os.listdir(directory_results)
    list_files = sorted([file for file in list_files if file.endswith('.csv')])
    sensor_1_x = []
    sensor_1_y = []
    sensor_1_z = []

    sensor_2_x = []
    sensor_2_y = []
    sensor_2_z = []

    sensor_3_x = []
    sensor_3_y = []
    sensor_3_z = []

    sensor_4_x = []
    sensor_4_y = []
    sensor_4_z = []

    for file in list_files:
        data = pd.read_csv(directory_results + file, skipinitialspace=True)
        sensor_1_x.append(data['sensor 1 x'].tolist())
        sensor_1_y.append(data['sensor 1 y'].tolist())
        sensor_1_z.append(data['sensor 1 z'].tolist())

        sensor_2_x.append(data['sensor 2 x'].tolist())
        sensor_2_y.append(data['sensor 2 y'].tolist())
        sensor_2_z.append(data['sensor 2 z'].tolist())

        sensor_3_x.append(data['sensor 3 x'].tolist())
        sensor_3_y.append(data['sensor 3 y'].tolist())
        sensor_3_z.append(data['sensor 3 z'].tolist())

        sensor_4_x.append(data['sensor 4 x'].tolist())
        sensor_4_y.append(data['sensor 4 y'].tolist())
        sensor_4_z.append(data['sensor 4 z'].tolist())

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    all_vals_x = []
    all_vals_y = []

    x_mins = []
    x_maxs = []
    y_mins = []
    y_maxs = []
    z_mins = []
    z_maxs = []

    for j in range(len(sensor_1_x)):
        ox, oy, oz = calculate_centroid_triangle(sensor_1_x[j], sensor_1_y[j], sensor_1_z[j],
                                                 sensor_2_x[j], sensor_2_z[j],
                                                 sensor_4_x[j], sensor_4_z[j])

        z_maxs.append(np.amin(sensor_4_z[j]))
        z_mins.append(np.amax(sensor_2_z[j]))

        x_maxs.append(np.amin(sensor_4_z[j]))
        x_mins.append(np.amax(sensor_2_z[j]))

        y_maxs.append(np.amin(sensor_4_y[j]))
        y_mins.append(np.amax(sensor_4_y[j]))

        ax.plot(sensor_1_x[j], sensor_1_y[j], sensor_1_z[j])  # label='sensor 1 trajectory ' + str(j))
        ax.plot(sensor_2_x[j], sensor_2_y[j], sensor_2_z[j])  # label='sensor 2 trajectory ' + str(j))
        ax.plot(sensor_3_x[j], sensor_3_y[j], sensor_3_z[j])  # label='sensor 3 trajectory ' + str(j))
        ax.plot(sensor_4_x[j], sensor_4_y[j], sensor_4_z[j])  # label='sensor 4 trajectory ' + str(j))

        all_vals_x.append(ox)
        all_vals_y.append(oy)
        ax.plot(oy, oz, ox, '-o', label='center point' + str(j))

    pooling = 5
    inner_radius = 8
    x_min = np.amin(x_mins)
    x_max = np.amax(x_maxs)
    y_min = np.amin(y_mins)
    y_max = np.amax(y_maxs)
    z_min = np.amin(z_mins)
    z_max = np.amax(z_maxs)

    x_cilinder = np.linspace(31, 47, 500)
    z = np.linspace(y_min - 7, y_max + 7, 500)
    Xc, Zc = np.meshgrid(x_cilinder, z)
    Yc1 = np.sqrt(inner_radius ** 2 - (39 - Xc) ** 2) - 180
    Yc2 = -np.sqrt(inner_radius ** 2 - (39 - Xc) ** 2) - 180

    # Draw parameters
    rstride = 10
    cstride = 20
    # plot the cylinders
    ax.plot_wireframe(Xc, Zc, Yc1, alpha=0.2, rstride=rstride, cstride=cstride, color='orange')
    ax.plot_wireframe(Xc, Zc, Yc2, alpha=0.2, rstride=rstride, cstride=cstride, color='orange')

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    # ax.set_xlim(55, 95)

    plt.figure()

    for j in range(len(all_vals_x)):
        plt.subplot(211)
        plt.plot(oz, all_vals_x[j], label='center point' + str(j))
        # plt.plot(sensor_3_y[j], sensor_3_z[j], 's')
        plt.subplot(212)
        plt.plot(oz, all_vals_y[j], label='center point' + str(j))
        # plt.plot(sensor_3_y[j], sensor_3_x[j], 's')

    average_x = calculate_average(all_vals_x)
    average_y = calculate_average(all_vals_y)
    plt.subplot(211)
    plt.plot(oz, average_x, '--', label='average')
    plt.subplot(212)
    plt.plot(oz, average_y, '--', label='average')
    plt.xlabel('y (mm)')
    plt.legend(loc='best')
    plt.show()


def analyze_centering_task(dir_folder):
    list_folders = os.listdir(dir_folder)
    jacobian_results = [folder for folder in list_folders if "jacobian" in folder]
    potential_field = [folder for folder in list_folders if "potential_field" in folder]

    list_sensor_1_x_jacobian = []
    list_sensor_1_y_jacobian = []
    list_sensor_1_z_jacobian = []

    list_sensor_2_x_jacobian = []
    list_sensor_2_y_jacobian = []
    list_sensor_2_z_jacobian = []

    list_sensor_3_x_jacobian = []
    list_sensor_3_y_jacobian = []
    list_sensor_3_z_jacobian = []

    average_J_x = []
    average_J_y = []
    average_J_z = []

    list_sensor_1_x_potential = []
    list_sensor_1_y_potential = []
    list_sensor_1_z_potential = []

    list_sensor_2_x_potential = []
    list_sensor_2_y_potential = []
    list_sensor_2_z_potential = []

    list_sensor_3_x_potential = []
    list_sensor_3_y_potential = []
    list_sensor_3_z_potential = []

    average_P_x = []
    average_P_y = []
    average_P_z = []

    calibration_data_x_point_1_j = []
    calibration_data_y_point_1_j = []
    calibration_data_z_point_1_j = []

    calibration_data_x_point_2_j = []
    calibration_data_y_point_2_j = []
    calibration_data_z_point_2_j = []

    calibration_data_x_point_3_j = []
    calibration_data_y_point_3_j = []
    calibration_data_z_point_3_j = []

    calibration_data_x_point_4_j = []
    calibration_data_y_point_4_j = []
    calibration_data_z_point_4_j = []

    calibration_data_x_point_1_p = []
    calibration_data_y_point_1_p = []
    calibration_data_z_point_1_p = []

    calibration_data_x_point_2_p = []
    calibration_data_y_point_2_p = []
    calibration_data_z_point_2_p = []

    calibration_data_x_point_3_p = []
    calibration_data_y_point_3_p = []
    calibration_data_z_point_3_p = []

    calibration_data_x_point_4_p = []
    calibration_data_y_point_4_p = []
    calibration_data_z_point_4_p = []

    for j, folder in enumerate(potential_field):
        j_son_file = [file for file in os.listdir(os.path.join(dir_folder, folder)) if file.endswith('.json')][0]
        calibration_points = dm.read_data_json(os.path.join(dir_folder, folder, j_son_file))

        calibration_data_x_point_1_p.append(calibration_points["limit_point_x"][:6])
        calibration_data_x_point_2_p.append(calibration_points["limit_point_x"][13:17])
        calibration_data_x_point_3_p.append(calibration_points["limit_point_x"][25:30])
        calibration_data_x_point_4_p.append(calibration_points["limit_point_x"][38:40])

        calibration_data_y_point_1_p.append(calibration_points["limit_point_y"][:6])
        calibration_data_y_point_2_p.append(calibration_points["limit_point_y"][13:18])
        calibration_data_y_point_3_p.append(calibration_points["limit_point_y"][25:30])
        calibration_data_y_point_4_p.append(calibration_points["limit_point_y"][38:40])

        calibration_data_z_point_1_p.append(calibration_points["limit_point_z"][:6])
        calibration_data_z_point_2_p.append(calibration_points["limit_point_z"][13:18])
        calibration_data_z_point_3_p.append(calibration_points["limit_point_z"][25:30])
        calibration_data_z_point_4_p.append(calibration_points["limit_point_z"][38:40])

        data = pd.read_csv(os.path.join(dir_folder, folder, 'experiment_' + folder[-16:] + '_.csv'))
        list_sensor_1_x_potential.append(data['sensor 1 x'].tolist())
        list_sensor_1_y_potential.append(data['sensor 1 y'].tolist())
        list_sensor_1_z_potential.append(data['sensor 1 z'].tolist())

        list_sensor_2_x_potential.append(data['sensor 2 x'].tolist())
        list_sensor_2_y_potential.append(data['sensor 2 y'].tolist())
        list_sensor_2_z_potential.append(data['sensor 2 z'].tolist())

        list_sensor_3_x_potential.append(data['sensor 3 x'].tolist())
        list_sensor_3_y_potential.append(data['sensor 3 y'].tolist())
        list_sensor_3_z_potential.append(data['sensor 3 z'].tolist())

    for j, list in enumerate(list_sensor_1_x_potential):
        average_x = []
        average_z = []
        average_y = []
        for i in range(len(list)):
            average_x.append((list_sensor_1_x_potential[j][i] + list_sensor_2_x_potential[j][i] +
                              list_sensor_3_x_potential[j][i]) / 3)
            average_y.append((list_sensor_1_y_potential[j][i] + list_sensor_2_y_potential[j][i] +
                              list_sensor_3_y_potential[j][i]) / 3)
            average_z.append((list_sensor_1_z_potential[j][i] + list_sensor_2_z_potential[j][i] +
                              list_sensor_3_z_potential[j][i]) / 3)

        average_P_x.append(average_x)
        average_P_y.append(average_y)
        average_P_z.append(average_z)

    for j, folder in enumerate(jacobian_results):
        j_son_file = [file for file in os.listdir(os.path.join(dir_folder, folder)) if file.endswith('.json')][0]
        calibration_points = dm.read_data_json(os.path.join(dir_folder, folder, j_son_file))

        calibration_data_x_point_1_j.append(calibration_points["limit_point_x"][:6])
        calibration_data_x_point_2_j.append(calibration_points["limit_point_x"][13:18])
        calibration_data_x_point_3_j.append(calibration_points["limit_point_x"][25:30])
        calibration_data_x_point_4_j.append(calibration_points["limit_point_x"][38:40])

        calibration_data_y_point_1_j.append(calibration_points["limit_point_y"][:6])
        calibration_data_y_point_2_j.append(calibration_points["limit_point_y"][13:18])
        calibration_data_y_point_3_j.append(calibration_points["limit_point_y"][25:30])
        calibration_data_y_point_4_j.append(calibration_points["limit_point_y"][38:40])

        calibration_data_z_point_1_j.append(calibration_points["limit_point_z"][:6])
        calibration_data_z_point_2_j.append(calibration_points["limit_point_z"][13:18])
        calibration_data_z_point_3_j.append(calibration_points["limit_point_z"][25:30])
        calibration_data_z_point_4_j.append(calibration_points["limit_point_z"][38:40])

        data = pd.read_csv(os.path.join(dir_folder, folder, 'experiment_' + folder[-16:] + '_.csv'))
        list_sensor_1_x_jacobian.append(data['sensor 1 x'].tolist())
        list_sensor_1_y_jacobian.append(data['sensor 1 y'].tolist())
        list_sensor_1_z_jacobian.append(data['sensor 1 z'].tolist())

        list_sensor_2_x_jacobian.append(data['sensor 2 x'].tolist())
        list_sensor_2_y_jacobian.append(data['sensor 2 y'].tolist())
        list_sensor_2_z_jacobian.append(data['sensor 2 z'].tolist())

        list_sensor_3_x_jacobian.append(data['sensor 3 x'].tolist())
        list_sensor_3_y_jacobian.append(data['sensor 3 y'].tolist())
        list_sensor_3_z_jacobian.append(data['sensor 3 z'].tolist())

    for j, list in enumerate(list_sensor_1_x_jacobian):
        average_jx = []
        average_jz = []
        average_jy = []
        for i in range(len(list)):
            average_jx.append(
                (list_sensor_1_x_jacobian[j][i] + list_sensor_2_x_jacobian[j][i] + list_sensor_3_x_jacobian[j][i]) / 3)
            average_jy.append(
                (list_sensor_1_y_jacobian[j][i] + list_sensor_2_y_jacobian[j][i] + list_sensor_3_y_jacobian[j][i]) / 3)
            average_jz.append(
                (list_sensor_1_z_jacobian[j][i] + list_sensor_2_z_jacobian[j][i] + list_sensor_3_z_jacobian[j][i]) / 3)

        average_J_x.append(average_jx)
        average_J_y.append(average_jy)
        average_J_z.append(average_jz)

    """ax = plt.axes(projection='3d')
    for j in range(len(calibration_data_x_point_1_j)):
        ax.scatter3D(calibration_data_x_point_1_j[j], calibration_data_y_point_1_j[j],
                     calibration_data_z_point_1_j[j], cmap='Reds')
        ax.scatter3D(average_P_x[j], average_P_y[j], average_P_z[j])

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()"""
    x = np.linspace(-40, 9, int((9 + 40) / 0.05) + 1, endpoint=True)
    plt.figure()
    kern = np.ones(4)
    plt.subplot(121)
    for j in range(len(average_J_z)):
        # plt.plot(np.mean(calibration_data_x_point_1_j[j]), np.mean(calibration_data_z_point_1_j[j]), 'ro')
        # plt.plot(np.mean(calibration_data_x_point_2_j[j]), np.mean(calibration_data_z_point_2_j[j]), 'bo')
        # plt.plot(np.mean(calibration_data_x_point_3_j[j]), np.mean(calibration_data_z_point_3_j[j]), 'go')
        # plt.plot(np.mean(calibration_data_x_point_4_j[j]), np.mean(calibration_data_z_point_4_j[j]), 'yo')
        center_x = (np.mean(calibration_data_x_point_4_j[j]) / 2 - np.mean(calibration_data_x_point_2_j[j]) / 2) + \
                   np.max([np.mean(calibration_data_x_point_4_j[j]), np.mean(calibration_data_x_point_2_j[j])]) + 5
        center_y = (np.mean(calibration_data_z_point_2_j[j]) / 2 - np.mean(calibration_data_z_point_1_j[j]) / 2) + \
                   np.min([np.mean(calibration_data_z_point_2_j[j]), np.mean(calibration_data_z_point_1_j[j])])
        plt.plot(center_x, center_y, '+')

        # plt.plot(average_J_x[j], average_J_z[j], '-*', label='Jacobian experiment' + str(j))
        plt.plot(np.convolve(np.array(average_J_x[j]), kern, mode='valid') / 4,
                 np.convolve(np.array(average_J_z[j]), kern, mode='valid') / 4,
                 '-*', label='Jacobian experiment ' + str(j))
        plt.plot(average_J_x[j][0], average_J_z[j][0], 'rX')

    plt.legend(loc='best')
    plt.subplot(122)
    for j in range(len(average_P_x)):
        # plt.plot(np.mean(calibration_data_x_point_1_p[j]), np.mean(calibration_data_z_point_1_p[j]), 'ro')
        # plt.plot(np.mean(calibration_data_x_point_2_p[j]), np.mean(calibration_data_z_point_2_p[j]), 'bo')
        # plt.plot(np.mean(calibration_data_x_point_3_p[j]), np.mean(calibration_data_z_point_3_p[j]), 'go')
        # plt.plot(np.mean(calibration_data_x_point_4_p[j]), np.mean(calibration_data_z_point_4_p[j]), 'yo')
        center_x = (np.mean(calibration_data_x_point_4_p[j]) / 2 - np.mean(calibration_data_x_point_2_p[j]) / 2) + \
                   np.max([np.mean(calibration_data_x_point_4_p[j]), np.mean(calibration_data_x_point_2_p[j])]) + 5
        center_y = (np.mean(calibration_data_z_point_2_p[j]) / 2 - np.mean(calibration_data_z_point_1_p[j]) / 2) + \
                   np.min([np.mean(calibration_data_z_point_2_p[j]), np.mean(calibration_data_z_point_1_p[j])])
        plt.plot(center_x, center_y, '+')
        # plt.plot(average_P_x[j], average_P_z[j], '-*', label='P. Field experiment' + str(j))
        plt.plot(np.convolve(np.array(average_P_x[j]), kern, mode='valid') / 4,
                 np.convolve(np.array(average_P_z[j]), kern, mode='valid') / 4,
                 '-*', label='P. Field experiment ' + str(j))
        plt.plot(average_P_x[j][0], average_P_z[j][0], 'rX')

    plt.legend(loc='best')


def mapping_time(list_time):
    list_new_time = []
    for time in list_time:
        new_time = []
        buffer_time = []
        for i in range(len(time)):
            buffer_time.append(datetime.strptime(time[i][2:], '%y-%m-%d %H:%M:%S.%f'))
        for i in range(len(buffer_time)):
            new_time.append((buffer_time[i] - buffer_time[0]).total_seconds())
        list_new_time.append(new_time)
    return list_new_time


def find_finish_time(lists_time_series):
    finish_time = []
    for list_time in lists_time_series:
        finish_time.append(list_time[-1])
    return finish_time


def calculate_stop_times(lists_stepper_state, offset=500):
    stop_frequency = []
    for list_state in lists_stepper_state:
        print(list_state)
        counts = np.bincount((np.array(list_state) + offset).astype(int))
        stop_frequency.append(len([x for x in counts if x > 5]))
    print('NUMBERS:', stop_frequency)
    return stop_frequency


def calculate_list_stat(data):
    average = sum(data) * 1.00 / len(data)
    std = np.array(data).std()
    return average, std


def analyze_navigation_task(dir_folder):
    list_folders = os.listdir(dir_folder)
    jacobian_results = [folder for folder in list_folders if "jacobian" in folder]
    potential_field = [folder for folder in list_folders if "potential_field" in folder]

    list_time_jacobian = []
    list_time_potential = []

    list_sensor_1_x_jacobian = []
    list_sensor_1_y_jacobian = []
    list_sensor_1_z_jacobian = []

    list_sensor_2_x_jacobian = []
    list_sensor_2_y_jacobian = []
    list_sensor_2_z_jacobian = []

    list_sensor_3_x_jacobian = []
    list_sensor_3_y_jacobian = []
    list_sensor_3_z_jacobian = []

    average_J_x = []
    average_J_y = []
    average_J_z = []

    list_sensor_1_x_potential = []
    list_sensor_1_y_potential = []
    list_sensor_1_z_potential = []

    list_sensor_2_x_potential = []
    list_sensor_2_y_potential = []
    list_sensor_2_z_potential = []

    list_sensor_3_x_potential = []
    list_sensor_3_y_potential = []
    list_sensor_3_z_potential = []

    average_P_x = []
    average_P_y = []
    average_P_z = []

    calibration_data_x_point_1_j = []
    calibration_data_y_point_1_j = []
    calibration_data_z_point_1_j = []

    calibration_data_x_point_2_j = []
    calibration_data_y_point_2_j = []
    calibration_data_z_point_2_j = []

    calibration_data_x_point_3_j = []
    calibration_data_y_point_3_j = []
    calibration_data_z_point_3_j = []

    calibration_data_x_point_4_j = []
    calibration_data_y_point_4_j = []
    calibration_data_z_point_4_j = []

    calibration_data_x_point_1_p = []
    calibration_data_y_point_1_p = []
    calibration_data_z_point_1_p = []

    calibration_data_x_point_2_p = []
    calibration_data_y_point_2_p = []
    calibration_data_z_point_2_p = []

    calibration_data_x_point_3_p = []
    calibration_data_y_point_3_p = []
    calibration_data_z_point_3_p = []

    calibration_data_x_point_4_p = []
    calibration_data_y_point_4_p = []
    calibration_data_z_point_4_p = []

    list_target_x_jacobian = []
    list_target_y_jacobian = []
    list_target_x_potential = []
    list_target_y_potential = []

    for j, folder in enumerate(potential_field):
        j_son_file = [file for file in os.listdir(os.path.join(dir_folder, folder)) if file.endswith('.json')][0]
        calibration_points = dm.read_data_json(os.path.join(dir_folder, folder, j_son_file))

        calibration_data_x_point_1_p.append(calibration_points["limit_point_x"][:6])
        calibration_data_x_point_2_p.append(calibration_points["limit_point_x"][13:17])
        calibration_data_x_point_3_p.append(calibration_points["limit_point_x"][25:30])
        calibration_data_x_point_4_p.append(calibration_points["limit_point_x"][38:40])

        calibration_data_y_point_1_p.append(calibration_points["limit_point_y"][:6])
        calibration_data_y_point_2_p.append(calibration_points["limit_point_y"][13:18])
        calibration_data_y_point_3_p.append(calibration_points["limit_point_y"][25:30])
        calibration_data_y_point_4_p.append(calibration_points["limit_point_y"][38:40])

        calibration_data_z_point_1_p.append(calibration_points["limit_point_z"][:6])
        calibration_data_z_point_2_p.append(calibration_points["limit_point_z"][13:18])
        calibration_data_z_point_3_p.append(calibration_points["limit_point_z"][25:30])
        calibration_data_z_point_4_p.append(calibration_points["limit_point_z"][38:40])

        data = pd.read_csv(os.path.join(dir_folder, folder, 'experiment_' + folder[-16:] + '_.csv'))

        list_target_x_potential.append(data['filtered x'].tolist())
        list_target_y_potential.append(data['filtered y'].tolist())

        list_sensor_1_x_potential.append(data['sensor 1 x'].tolist())
        list_sensor_1_y_potential.append(data['sensor 1 y'].tolist())
        list_sensor_1_z_potential.append(data['sensor 1 z'].tolist())

        list_sensor_2_x_potential.append(data['sensor 2 x'].tolist())
        list_sensor_2_y_potential.append(data['sensor 2 y'].tolist())
        list_sensor_2_z_potential.append(data['sensor 2 z'].tolist())

        list_sensor_3_x_potential.append(data['sensor 3 x'].tolist())
        list_sensor_3_y_potential.append(data['sensor 3 y'].tolist())
        list_sensor_3_z_potential.append(data['sensor 3 z'].tolist())

        list_time_potential.append(data['time'].tolist())

    for j, list in enumerate(list_sensor_1_x_potential):
        average_x = []
        average_z = []
        average_y = []
        for i in range(len(list)):
            average_x.append((list_sensor_1_x_potential[j][i] + list_sensor_2_x_potential[j][i] +
                              list_sensor_3_x_potential[j][i]) / 3)
            average_y.append((list_sensor_1_y_potential[j][i] + list_sensor_2_y_potential[j][i] +
                              list_sensor_3_y_potential[j][i]) / 3)
            average_z.append((list_sensor_1_z_potential[j][i] + list_sensor_2_z_potential[j][i] +
                              list_sensor_3_z_potential[j][i]) / 3)

        average_P_x.append(average_x)
        average_P_y.append(average_y)
        average_P_z.append(average_z)

    for j, folder in enumerate(jacobian_results):
        j_son_file = [file for file in os.listdir(os.path.join(dir_folder, folder)) if file.endswith('.json')][0]
        calibration_points = dm.read_data_json(os.path.join(dir_folder, folder, j_son_file))

        calibration_data_x_point_1_j.append(calibration_points["limit_point_x"][:6])
        calibration_data_x_point_2_j.append(calibration_points["limit_point_x"][13:18])
        calibration_data_x_point_3_j.append(calibration_points["limit_point_x"][25:30])
        calibration_data_x_point_4_j.append(calibration_points["limit_point_x"][38:40])

        calibration_data_y_point_1_j.append(calibration_points["limit_point_y"][:6])
        calibration_data_y_point_2_j.append(calibration_points["limit_point_y"][13:18])
        calibration_data_y_point_3_j.append(calibration_points["limit_point_y"][25:30])
        calibration_data_y_point_4_j.append(calibration_points["limit_point_y"][38:40])

        calibration_data_z_point_1_j.append(calibration_points["limit_point_z"][:6])
        calibration_data_z_point_2_j.append(calibration_points["limit_point_z"][13:18])
        calibration_data_z_point_3_j.append(calibration_points["limit_point_z"][25:30])
        calibration_data_z_point_4_j.append(calibration_points["limit_point_z"][38:40])

        data = pd.read_csv(os.path.join(dir_folder, folder, 'experiment_' + folder[-16:] + '_.csv'))

        list_target_x_jacobian.append(data['filtered x'].tolist())
        list_target_y_jacobian.append(data['filtered y'].tolist())

        list_sensor_1_x_jacobian.append(data['sensor 1 x'].tolist())
        list_sensor_1_y_jacobian.append(data['sensor 1 y'].tolist())
        list_sensor_1_z_jacobian.append(data['sensor 1 z'].tolist())

        list_sensor_2_x_jacobian.append(data['sensor 2 x'].tolist())
        list_sensor_2_y_jacobian.append(data['sensor 2 y'].tolist())
        list_sensor_2_z_jacobian.append(data['sensor 2 z'].tolist())

        list_sensor_3_x_jacobian.append(data['sensor 3 x'].tolist())
        list_sensor_3_y_jacobian.append(data['sensor 3 y'].tolist())
        list_sensor_3_z_jacobian.append(data['sensor 3 z'].tolist())

        list_time_jacobian.append(data['time'].tolist())

    for j, list in enumerate(list_sensor_1_x_jacobian):
        average_jx = []
        average_jz = []
        average_jy = []
        for i in range(len(list)):
            average_jx.append(
                (list_sensor_1_x_jacobian[j][i] + list_sensor_2_x_jacobian[j][i] + list_sensor_3_x_jacobian[j][i]) / 3)
            average_jy.append(
                (list_sensor_1_y_jacobian[j][i] + list_sensor_2_y_jacobian[j][i] + list_sensor_3_y_jacobian[j][i]) / 3)
            average_jz.append(
                (list_sensor_1_z_jacobian[j][i] + list_sensor_2_z_jacobian[j][i] + list_sensor_3_z_jacobian[j][i]) / 3)

        average_J_x.append(average_jx)
        average_J_y.append(average_jy)
        average_J_z.append(average_jz)

    plt.figure()
    ax = plt.axes(projection='3d')
    sampling_per_number = 3
    # for j in range(len(calibration_data_x_point_1_j)):
    #     ax.scatter3D(calibration_data_x_point_1_j[j][::sampling_per_number], calibration_data_y_point_1_j[j][::sampling_per_number],
    #                  calibration_data_z_point_1_j[j][::sampling_per_number], cmap='Reds')
    #     ax.scatter3D(calibration_data_x_point_2_j[j][::sampling_per_number], calibration_data_y_point_2_j[j][::sampling_per_number],
    #                  calibration_data_z_point_2_j[j][::sampling_per_number], cmap='Reds')
    #     ax.scatter3D(calibration_data_x_point_3_j[j][::sampling_per_number], calibration_data_y_point_3_j[j][::sampling_per_number],
    #                  calibration_data_z_point_3_j[j][::sampling_per_number], cmap='Reds')
    #     ax.scatter3D(calibration_data_x_point_4_j[j][::sampling_per_number], calibration_data_y_point_4_j[j][::sampling_per_number],
    #                  calibration_data_z_point_4_j[j][::sampling_per_number], cmap='Reds')
    #     ax.scatter3D(average_J_x[j][::sampling_per_number], average_J_y[j][::sampling_per_number], average_J_z[j][::sampling_per_number])
    j = 8
    print(average_J_x[j][::sampling_per_number])
    ax.scatter3D(calibration_data_x_point_1_j[j][::sampling_per_number],
                 calibration_data_y_point_1_j[j][::sampling_per_number],
                 calibration_data_z_point_1_j[j][::sampling_per_number], cmap='Reds')
    ax.scatter3D(calibration_data_x_point_2_j[j][::sampling_per_number],
                 calibration_data_y_point_2_j[j][::sampling_per_number],
                 calibration_data_z_point_2_j[j][::sampling_per_number], cmap='Reds')
    ax.scatter3D(calibration_data_x_point_3_j[j][::sampling_per_number],
                 calibration_data_y_point_3_j[j][::sampling_per_number],
                 calibration_data_z_point_3_j[j][::sampling_per_number], cmap='Reds')
    ax.scatter3D(calibration_data_x_point_4_j[j][::sampling_per_number],
                 calibration_data_y_point_4_j[j][::sampling_per_number],
                 calibration_data_z_point_4_j[j][::sampling_per_number], cmap='Reds')
    ax.scatter3D(average_J_x[j][::sampling_per_number], average_J_y[j][::sampling_per_number],
                 average_J_z[j][::sampling_per_number])

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.figure()
    ax2 = plt.axes(projection='3d')
    # for j in range(len(calibration_data_x_point_1_j)):
    #     ax2.scatter3D(calibration_data_x_point_1_p[j][::sampling_per_number], calibration_data_y_point_1_p[j][::sampling_per_number],
    #                   calibration_data_z_point_1_p[j][::sampling_per_number], cmap='Reds')
    #     # ax.scatter3D(calibration_data_x_point_2_p[j], calibration_data_y_point_2_p[j],
    #     #             calibration_data_z_point_2_p[j], cmap='Reds')
    #     ax2.scatter3D(calibration_data_x_point_3_p[j][::sampling_per_number], calibration_data_y_point_3_p[j][::sampling_per_number],
    #                   calibration_data_z_point_3_p[j][::sampling_per_number], cmap='Reds')
    #     ax2.scatter3D(calibration_data_x_point_4_p[j][::sampling_per_number], calibration_data_y_point_4_p[j][::sampling_per_number],
    #                   calibration_data_z_point_4_p[j][::sampling_per_number], cmap='Reds')
    #     ax2.scatter3D(average_P_x[j][::sampling_per_number], average_P_y[j][::sampling_per_number], average_P_z[j][::sampling_per_number])
    #
    # ax2.set_xlabel('X')
    # ax2.set_ylabel('Y')
    # ax2.set_zlabel('Z')
    # plt.show()

    plt.figure(3)
    for j in range(len(average_J_z)):
        center_x = (np.mean(calibration_data_x_point_4_j[j]) / 2 - np.mean(calibration_data_x_point_2_j[j]) / 2) + \
                   np.max([np.mean(calibration_data_x_point_4_j[j]), np.mean(calibration_data_x_point_2_j[j])]) + 5
        center_y = (np.mean(calibration_data_z_point_2_j[j]) / 2 - np.mean(calibration_data_z_point_1_j[j]) / 2) + \
                   np.min([np.mean(calibration_data_z_point_2_j[j]), np.mean(calibration_data_z_point_1_j[j])])
        print(center_x, center_y, 'jacobian')
        path_x = np.ones(len(average_J_y[j])) * center_x
        path_y = np.ones(len(average_J_y[j])) * center_y
        # plt.subplot(221)
        # plt.plot(average_J_y[j], average_J_z[j], '-*', label='Jacobian experiment' + str(j))
        # plt.subplot(222)
        # plt.plot(average_J_y[j], average_J_x[j], '-*', label='Jacobian experiment' + str(j))
        plt.subplot(221)
        plt.plot(average_J_y[j], np.absolute(average_J_z[j] - path_y), '-o')
        plt.subplot(222)
        plt.plot(average_J_y[j], np.absolute(average_J_x[j] - path_x), '-o')

    plt.subplot(221)
    plt.gca().set_title('Error X Axis')
    plt.ylabel('Jacobian')
    plt.subplot(222)
    plt.gca().set_title('Error Y Axis')
    plt.subplot(223)
    plt.ylabel('P. Field')

    for j in range(len(average_P_x)):
        center_x = (np.mean(calibration_data_x_point_4_p[j]) / 2 - np.mean(calibration_data_x_point_2_p[j]) / 2) + \
                   np.max([np.mean(calibration_data_x_point_4_p[j]), np.mean(calibration_data_x_point_2_p[j])]) + 5
        center_y = (np.mean(calibration_data_z_point_2_p[j]) / 2 - np.mean(calibration_data_z_point_1_p[j]) / 2) + \
                   np.min([np.mean(calibration_data_z_point_2_p[j]), np.mean(calibration_data_z_point_1_p[j])])
        print(center_x, center_y)
        path_x = np.ones(len(average_P_y[j])) * center_x
        path_y = np.ones(len(average_P_y[j])) * center_y
        # plt.subplot(221)
        # plt.plot(average_P_y[j], average_P_z[j], '-*', label='P. Field experiment' + str(j))
        # plt.subplot(222)
        # plt.plot(average_P_y[j], average_P_x[j], '-*', label='P. Field experiment' + str(j))
        plt.subplot(223)
        plt.plot(average_P_y[j], np.absolute(average_P_z[j] - path_y), '-o')
        plt.subplot(224)
        plt.plot(average_P_y[j], np.absolute(average_P_x[j] - path_x), '-o')

    plt.figure(4)
    for j in range(len(list_target_x_jacobian)):
        center_point_x_j = np.ones(len(average_J_y[j])) * 150
        center_point_y_j = np.ones(len(average_J_y[j])) * 150

        plt.subplot(221)
        plt.plot(average_J_y[j], np.abs(list_target_x_jacobian[j] - center_point_x_j))
        plt.subplot(222)
        plt.plot(average_J_y[j], np.abs(list_target_y_jacobian[j] - center_point_y_j))

    for j in range(len(list_target_y_potential)):
        center_point_x_j = np.ones(len(average_P_y[j])) * 150
        center_point_y_j = np.ones(len(average_P_y[j])) * 150

        plt.subplot(223)
        plt.plot(average_P_y[j], np.abs(list_target_x_potential[j] - center_point_x_j))
        plt.subplot(224)
        plt.plot(average_P_y[j], np.abs(list_target_y_potential[j] - center_point_y_j))

    plt.subplot(221)
    plt.gca().set_title('Error X Axis')
    plt.ylabel('Jacobian')
    plt.subplot(222)
    plt.gca().set_title('Error Y Axis')
    plt.subplot(223)
    plt.ylabel('P. Field')

    j = 3

    plt.figure(5, figsize=(7, 7), frameon=False)
    plt.subplot(111)
    plt.plot(average_J_y[j][::sampling_per_number], average_J_z[j][::sampling_per_number], 'ro')
    plt.xlim(-25, 125)
    plt.ylim(-200, -50)
    plt.savefig('C:\\Users\\cflai\\Documents\\TROspecial\\result_figure\\' + 'test.png', transparent=True)

    i = 9
    gt = GroundTruth(np.array(average_J_y[i][::sampling_per_number]), length=35)
    # gt = GroundTruthSPath(np.array(average_J_y[i][::sampling_per_number]), length1=1)
    gt.x_init()
    gt.y_init(x_flip=True, y_flip=False)  # path 2
    # gt.y_init(x_flip=True, y_flip=True)  # path 3
    # gt.y_init(x_flip=True, y_flip=False)  # path 4
    gt.shift_y(dy=-108.72)  # path 2
    # gt.shift_y(dy=-139)  # path 3
    # gt.shift_y(dy=-102.5)  # path 4
    print("AUC:", gt.diff_area(average_J_z[i][::sampling_per_number]))
    plt.figure(i + 7, figsize=(7, 7), frameon=False)
    plt.subplot(111)
    plt.plot(average_J_y[i][::sampling_per_number], average_J_z[i][::sampling_per_number], 'yx')
    plt.plot(average_J_y[i][::sampling_per_number], gt.y, 'ro')
    plt.fill_between(average_J_y[i][::sampling_per_number],
                     average_J_z[i][::sampling_per_number],
                     gt.y,
                     alpha=0.5)

    # plt.xlim(-25, 125)
    # plt.ylim(-200, -50)

    # Finish time session:
    list_time_delta_jacobian = mapping_time(list_time_jacobian)
    list_time_delta_potential = mapping_time(list_time_potential)
    finish_time_jacobian = find_finish_time(list_time_delta_jacobian)
    finish_time_potential = find_finish_time(list_time_delta_potential)
    average_finish_time_jacobian = calculate_average(finish_time_jacobian)
    average_finish_time_potential = calculate_average(finish_time_potential)
    print('jacobian finish time:', average_finish_time_jacobian)
    print('potential finish time:', average_finish_time_potential)
    print('------------p value-------------')
    calculate_kruskal_p_value(finish_time_jacobian, finish_time_potential)

    return


# end of analyze_navigation_task

def analyze_navigation_task_stop_frequency(dir_folder):
    list_folders = os.listdir(dir_folder)
    jacobian_results = [folder for folder in list_folders if "jacobian" in folder]
    potential_field = [folder for folder in list_folders if "potential_field" in folder]

    list_time_jacobian = []
    list_stepper_jacobian = []
    list_time_potential = []
    list_stepper_potential = []

    for j, folder in enumerate(jacobian_results):
        data = pd.read_csv(os.path.join(dir_folder, folder, 'experiment_' + folder[-16:] + '_.csv'))
        list_time_jacobian.append(data['time'].tolist())
        list_stepper_jacobian.append(data['stepper'].tolist())

    for j, folder in enumerate(potential_field):
        data = pd.read_csv(os.path.join(dir_folder, folder, 'experiment_' + folder[-16:] + '_.csv'))
        list_time_potential.append(data['time'].tolist())
        list_stepper_potential.append(data['stepper'].tolist())

    for j in range(len(list_stepper_jacobian)):
        plt.subplot(211)
        plt.plot(list_stepper_jacobian[j], 'o', label='Jacobian experiment ' + str(j))
        plt.legend(loc='best')
        plt.title('Stepper_Jacobian')

    for j in range(len(list_stepper_potential)):
        plt.subplot(212)
        plt.plot(list_stepper_potential[j], 'o', label='Potential experiment ' + str(j))
        plt.legend(loc='best')
        plt.title('Stepper_Potential')

    jacobian_stop_frequency = calculate_stop_times(list_stepper_jacobian)
    stat_jacobian_stop_frequency = calculate_list_stat(jacobian_stop_frequency)
    potential_stop_frequency = calculate_stop_times(list_stepper_potential)
    stat_potential_stop_frequency = calculate_list_stat(potential_stop_frequency)
    print('Jacobian(Avg, Std):', stat_jacobian_stop_frequency, 'Potential(Avg, Std):', stat_potential_stop_frequency)
    print('------------p value-------------')
    calculate_kruskal_p_value(stat_jacobian_stop_frequency, stat_potential_stop_frequency)


def canalyze_centering_img(dir_folder):
    list_folders = os.listdir(dir_folder)
    jacobian_results = [folder for folder in list_folders if "jacobian" in folder]
    potential_field = [folder for folder in list_folders if "potential_field" in folder]

    list_target_x_jacobian = []
    list_target_y_jacobian = []

    list_target_x_potential = []
    list_target_y_potential = []

    for j, folder in enumerate(potential_field):
        j_son_file = [file for file in os.listdir(os.path.join(dir_folder, folder)) if file.endswith('.json')][0]
        data = pd.read_csv(os.path.join(dir_folder, folder, 'experiment_' + folder[-16:] + '_.csv'))
        list_target_x_jacobian.append(data['filtered x'].tolist())
        list_target_y_jacobian.append(data['filtered y'].tolist())

    for j, folder in enumerate(jacobian_results):
        j_son_file = [file for file in os.listdir(os.path.join(dir_folder, folder)) if file.endswith('.json')][0]

        data = pd.read_csv(os.path.join(dir_folder, folder, 'experiment_' + folder[-16:] + '_.csv'))
        list_target_x_potential.append(data['filtered x'].tolist())
        list_target_y_potential.append(data['filtered y'].tolist())

    color_list = ['']
    plt.figure()

    for j in range(len(list_target_x_jacobian)):
        plt.subplot(231)
        plt.plot(list_target_x_jacobian[j],
                 list_target_y_jacobian[j],
                 marker='-*', color=color_list[j], label='Jacobian experiment ' + str(j))
        plt.legend(loc='best')
        center_target = np.ones(len(list_target_x_jacobian[j])) * 150
        plt.plot(150, 150, 'rX')

        plt.subplot(232)
        plt.plot(list_target_x_jacobian[j] - center_target, marker='None')
        plt.subplot(233)
        plt.plot(list_target_y_jacobian[j] - center_target, marker='None')

    for j in range(len(list_target_x_potential)):
        plt.subplot(234)
        plt.plot(list_target_x_potential[j],
                 list_target_y_potential[j],
                 '-*', label='P. Field experiment ' + str(j))

        plt.plot(150, 150, 'rX')
        plt.legend(loc='best')
        center_target = np.ones(len(list_target_x_potential[j])) * 150
        plt.plot(150, 150, 'rX')

        plt.subplot(235)
        plt.plot(list_target_x_potential[j] - center_target, marker='None')
        plt.subplot(236)
        plt.plot(list_target_y_potential[j] - center_target, marker='None')


if __name__ == '__main__':
    # plot 3_D data
    # directory_2 = os.getcwd() + '/results/n_control/straight_line/'
    # analyze_results(directory)203
    # directory_1 = os.getcwd() + '/data/calibration/gt_trajectories/straight_line/'
    # plot_3D_data(directory_1)
    directory = os.getcwd() + '/to_analyze/task_2/path_2/'
    analyze_navigation_task(directory)
    # analyze_navigation_task_stop_frequency(directory)
    plt.show()
