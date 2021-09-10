import os
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import glob
import data_managament as dm
import scipy.stats as stats
from scipy.stats import kruskal
from datetime import datetime
import seaborn as sns
from scipy.signal import find_peaks, peak_prominences
import trajectories
import shutil
import cv2



def smooth(a, WSZ=5):
    # a: NumPy 1-D array containing the data to be smoothed
    # WSZ: smoothing window size needs, which must be odd number,
    # as in the original MATLAB implementation
    out0 = np.convolve(a, np.ones(WSZ, dtype=int), 'valid')/WSZ
    r = np.arange(1, WSZ-1, 2)
    start = np.cumsum(a[:WSZ-1])[::2]/r
    stop = (np.cumsum(a[:-WSZ:-1])[::2]/r)[::-1]
    return np.concatenate((start, out0, stop))


def check_nan(array):
    return [element for element in array if not(np.isnan(element))]


def determine_vertex_and_edges(arrays_points, plot='False'):

    """
    Function to obtain separately the 8 points from a
    @param arrays_points: an array of dictionaries with arrays of points which contains the keys:
    'limit_point_x', 'limit_point_y', 'limit_point_z'
    @param plot: (bool) True if you want to visualize the points, otherwise False by default
    @return:
    """

    original_x = []
    original_y = []
    original_z = []
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'olive', 'cyan', 'pink']
    # initialize empty arrays to save each of the points separate by coordinate
    points_x = [[] for _ in range(8)]
    points_y = [[] for _ in range(8)]
    points_z = [[] for _ in range(8)]

    for array_points in arrays_points:
        raw_x = check_nan(array_points["limit_point_x"])
        raw_y = check_nan(array_points["limit_point_y"])
        raw_z = check_nan(array_points["limit_point_z"])

        if raw_x:
            original_x.append(raw_x)
            original_y.append(raw_y)
            original_z.append(raw_z)


    for k in range(len(original_x)):
        # collecting the point 0
        # 2DO find a better way to add it in the loop
        points_x[0] = points_x[0] + original_x[k][:6]
        points_y[0] = points_y[0] + original_y[k][:6]
        points_z[0] = points_z[0] + original_z[k][:6]

        # collecting the points separately by coordinate 
        for j in range(1, 8):
            starting_index = j * 6 + 1
            ending_index = j * 6 + 6
            points_x[j] = points_x[j] + original_x[k][starting_index: ending_index]
            points_y[j] = points_y[j] + original_y[k][starting_index: ending_index]
            points_z[j] = points_z[j] + original_z[k][starting_index: ending_index]

    if plot is True:
        fig = plt.figure()
        ax1 = fig.add_subplot(231)
        ax2 = fig.add_subplot(232)
        ax3 = fig.add_subplot(233)
        ax4 = fig.add_subplot(234)
        ax5 = fig.add_subplot(235)
        ax6 = fig.add_subplot(236)

        ax1.title.set_text('x vs y')
        ax2.title.set_text('x vs z')
        ax3.title.set_text('y vs z')

        # Surface 1 (front)
        ax1.plot(points_x[0], points_z[0], 'o', label='0')
        ax1.plot(points_x[2], points_z[2], 'o', label='2')
        ax1.plot(points_x[4], points_z[4], 'o', label='4')
        ax1.plot(points_x[6], points_z[6], 'o', label='6')

        ax1.legend(loc='best')

        # Surface 2 (right side)
        ax2.plot(points_y[0], points_x[0], 'o', label='0')
        ax2.plot(points_y[1], points_x[1], 'o', label='1')
        ax2.plot(points_y[4], points_x[4], 'o', label='4')
        ax2.plot(points_y[5], points_x[5], 'o', label='5')

        ax2.legend(loc='best')

        # Surface 3 (lower)
        for i in range(4):
            ax3.plot(points_y[i], points_z[i], 'o', label=str(i))
            ax3.plot(np.median(points_y[i]), np.median(points_z[i]), 'y*')

        ax3.legend(loc='best')

        # Surface 4 (back)

        ax4.plot(points_x[1], points_z[1], 'o', label='1')
        ax4.plot(points_x[3], points_z[3], 'o', label='3')
        ax4.plot(points_x[5], points_z[5], 'o', label='5')
        ax4.plot(points_x[7], points_z[7], 'o', label='7')

        ax4.legend(loc='best')

        # Surface 5 (left side)

        ax5.plot(points_y[2], points_x[2], 'o', label='2')
        ax5.plot(points_y[3], points_x[3], 'o', label='3')
        ax5.plot(points_y[6], points_x[5], 'o', label='6')
        ax5.plot(points_y[7], points_x[7], 'o', label='7')

        ax5.legend(loc='best')

        # Surface 6 (upper)
        for i in range(4, 8):
            ax6.plot(points_y[i], points_z[i], 'o', label=str(i))
            ax6.plot(np.median(points_y[i]), np.median(points_z[i]), 'y*')

        ax6.legend(loc='best')

    #for j in range(8):
    #    print('Point ', j)
    #    print('x', points_x[j])
    #    print('y', points_y[j])
    #    print('z', points_z[j])

    return points_x, points_y, points_z


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
    num_points = (int(max_value)+1 - (int(min_value)-1)) * 2
    discretised_series = np.linspace(int(min_value)-1, int(max_value)+1, num_points, endpoint=False)
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
            discreet_y[j] = discreet_y[j-1]
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
            discreet_series[j] = discreet_series[j-1]

    return discreet_series, error_sries


def build_ground_truth_paths(directory_files, plot=True):

    """
    A function to build the ground truth trajectories
    @param directory_files: directory where the experimental data is storaged
    @param plot:
    @return:
    """

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

    line_x = list(range(int(min_x)-2, int(max_x)+2))
    line_y = np.ones((len(line_x)))*(min_y + (max_y-min_y)*0.5)
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
        #for j in range(len(list_all_z)):
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

        plt.ylim([min_y-30, max_y+30])
            #plt.axis('off')
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

    x_line = np.ones(500)*76
    z_line = np.ones(500)*-158
    y_line = np.linspace(-15, 140, 500)

    # Draw parameters
    rstride = 10
    cstride = 20
    # plot the cylinders
    ax.plot_wireframe(Xc, Zc, Yc1, alpha=0.2, rstride=rstride, cstride=cstride, color='orange')
    ax.plot_wireframe(Xc, Zc, Yc2, alpha=0.2, rstride=rstride, cstride=cstride, color='orange')
    ax.plot_wireframe(Xc2, Zc2, Yc3, alpha=0.1, rstride=rstride, cstride=cstride, color='orange' )
    ax.plot_wireframe(Xc2, Zc2, Yc4, alpha=0.1, rstride=rstride, cstride=cstride, color='orange')

    ax.plot(x_line, y_line, z_line, 'r-o')
    for j in range(len(list_all_x)):
        ax.plot(list_all_x[j], list_all_y[j], list_all_z[j], '-o', label='trajectory ' +str(j))
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
            discreet_list[j] = discreet_list[j-1]

    mean_y = [np.mean(sub_list) for sub_list in discreet_list]
    abs_error_list = [(np.abs(value-ground_truth[j])) for j, value in enumerate(mean_y)]

    return abs_error_list


def calcualte_error_dependant_list(ground_truth, measured, auxiliar_list):

    ground_truth = list(ground_truth)
    discreet_list = [[] for _ in range(len(ground_truth))]

    for index_point, point in enumerate(measured):
        for point_l in ground_truth:
            #print(point_l, point)
            if point_l <= point < point_l + 5:
                index_discreet = (ground_truth.index(point_l))
                discreet_list[index_discreet].append(measured[index_point])

    for j, sub_list in enumerate(discreet_list):
        if sub_list == []:
            discreet_list[j] = discreet_list[j-1]

    mean_y = [np.mean(sub_list) for sub_list in discreet_list]
    abs_error_list = [(np.abs(value-ground_truth[j])) for j, value in enumerate(mean_y)]
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


    up_error = [mean_value + stdv_y[j]*2 for j, mean_value in enumerate(mean_y)]
    down_error = [mean_value - stdv_y[j]*2 for j, mean_value in enumerate(mean_y)]

    line_x = np.linspace(int(min_x)-2, int(max_x)+2, len(mean_y))
    line_y = np.ones((len(line_x)))*(min_y + (max_y-min_y)*0.5)
    line_z = np.ones((len(line_x)))*(min_z + (max_z-min_z)*0.5)

    print(np.shape(line_x))
    print(np.shape(line_y))

    upper_lim_1 = [line + inner_radius for line in mean_y]
    upper_lim_2 = [line + outer_radius for line in mean_y]
    down_lim_1 = [line - inner_radius for line in mean_y]
    down_lim_2 = [line - outer_radius for line in mean_y]

    x_curve = np.linspace(discrete_series_x[-1], discrete_series_x[-1] + inner_radius, 180)
    x_curve_2 = np.linspace(discrete_series_x[-1], discrete_series_x[-1] + outer_radius, 180)

    y_curve = [np.sqrt(inner_radius**2-(x-discrete_series_x[-1])**2)+mean_y[-1] for x in x_curve]
    y_curve_2 = [-np.sqrt(inner_radius ** 2 - (x - discrete_series_x[-1]) ** 2) + mean_y[-1] for x in x_curve]

    y_curve_3 = [np.sqrt(outer_radius**2-(x-discrete_series_x[-1])**2)+mean_y[-1] for x in x_curve_2]
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

        #plt.plot(line_x, line_y, 'r--')
        plt.plot(discrete_series_x, down_lim_1, color='darkorange',  linewidth=3)
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
        #plt.title('Error Y axis')
        for j in range(len(list_all_z)):
            line_temp = list_all_y[j][::-1]
            plot_list = calcualte_error_list(discrete_series_x, line_temp)
            plt.plot(plot_list, 'o-', label='trajectory '+str(j))
            plt.ylabel('Error Y (mm)')
            plt.legend(loc='best')
            plt.xlabel('t')

        plt.figure()
        # plt.title('Error Y axis')
        for j in range(len(list_all_z)):
            line_temp = list_all_x[j]#[::-1]
            #plot_list = [np.abs(x - line_z[0]) for x in line_temp]
            plot_list = calcualte_error_dependant_list(mean_z, line_temp, discrete_series_x)
            plt.plot(discrete_series_x, plot_list,  'o-', label='trajectory '+str(j))
            plt.ylabel('Absolute error X (mm)')
            plt.legend(loc='best')
            plt.xlim(max(discrete_series_x), min(discrete_series_x))
            plt.xlabel('y (mm)')

        plt.figure()
        # plt.title('Error Y axis')
        for j in range(len(list_all_z)):
            line_temp = list_all_z[j]#[::-1]
            #plot_list = [np.abs(x - line_y[0]) for x in line_temp]
            plot_list = calcualte_error_dependant_list(mean_y, line_temp, discrete_series_x)
            plt.plot(discrete_series_x, plot_list, 'o-', label='trajectory '+str(j))
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
    #list_results_files = sorted([file for file in list_results_files if file.endswith('.csv')])
    #results_csv = sorted([file for file in results_csv if file.endswith('.csv')])
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
            #plt.plot(list_all_y[j], list_all_z[j], 'o-')
            #plt.ylim([-250, 150])
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
        oy.append((edge_1_y[j] + edge_2_y[j] + edge_3_y[j])/3)
        ox.append((edge_1_x[j] + edge_2_x[j] + edge_3_x[j])/3)

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

        ax.plot(sensor_1_x[j], sensor_1_y[j], sensor_1_z[j]) #label='sensor 1 trajectory ' + str(j))
        ax.plot(sensor_2_x[j], sensor_2_y[j], sensor_2_z[j]) #label='sensor 2 trajectory ' + str(j))
        ax.plot(sensor_3_x[j], sensor_3_y[j], sensor_3_z[j]) #label='sensor 3 trajectory ' + str(j))
        ax.plot(sensor_4_x[j], sensor_4_y[j], sensor_4_z[j]) #label='sensor 4 trajectory ' + str(j))

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
    #ax.set_xlim(55, 95)

    plt.figure()

    for j in range(len(all_vals_x)):
        plt.subplot(211)
        plt.plot(oz, all_vals_x[j], label='center point' + str(j))
        #plt.plot(sensor_3_y[j], sensor_3_z[j], 's')
        plt.subplot(212)
        plt.plot(oz, all_vals_y[j], label='center point' + str(j))
        #plt.plot(sensor_3_y[j], sensor_3_x[j], 's')


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
        list_sensor_1_y_potential .append(data['sensor 1 y'].tolist())
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
            average_x.append((list_sensor_1_x_potential[j][i] + list_sensor_2_x_potential[j][i] + list_sensor_3_x_potential[j][i])/3)
            average_y.append((list_sensor_1_y_potential[j][i] + list_sensor_2_y_potential[j][i] + list_sensor_3_y_potential[j][i])/3)
            average_z.append((list_sensor_1_z_potential[j][i] + list_sensor_2_z_potential[j][i] + list_sensor_3_z_potential[j][i])/3)

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
        list_sensor_1_y_jacobian .append(data['sensor 1 y'].tolist())
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
            average_jx.append((list_sensor_1_x_jacobian[j][i] + list_sensor_2_x_jacobian[j][i] + list_sensor_3_x_jacobian[j][i])/3)
            average_jy.append((list_sensor_1_y_jacobian[j][i] + list_sensor_2_y_jacobian[j][i] + list_sensor_3_y_jacobian[j][i])/3)
            average_jz.append((list_sensor_1_z_jacobian[j][i] + list_sensor_2_z_jacobian[j][i] + list_sensor_3_z_jacobian[j][i])/3)

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
    x = np.linspace(-40, 9, int((9+40)/0.05)+1, endpoint=True)
    plt.figure()
    kern = np.ones(4)
    plt.subplot(121)
    for j in range(len(average_J_z)):
        #plt.plot(np.mean(calibration_data_x_point_1_j[j]), np.mean(calibration_data_z_point_1_j[j]), 'ro')
        #plt.plot(np.mean(calibration_data_x_point_2_j[j]), np.mean(calibration_data_z_point_2_j[j]), 'bo')
        #plt.plot(np.mean(calibration_data_x_point_3_j[j]), np.mean(calibration_data_z_point_3_j[j]), 'go')
        #plt.plot(np.mean(calibration_data_x_point_4_j[j]), np.mean(calibration_data_z_point_4_j[j]), 'yo')
        center_x = (np.mean(calibration_data_x_point_4_j[j])/2 - np.mean(calibration_data_x_point_2_j[j])/2) + \
                   np.max([np.mean(calibration_data_x_point_4_j[j]), np.mean(calibration_data_x_point_2_j[j])]) + 5
        center_y = (np.mean(calibration_data_z_point_2_j[j])/2 - np.mean(calibration_data_z_point_1_j[j])/2) + \
                   np.min([np.mean(calibration_data_z_point_2_j[j]), np.mean(calibration_data_z_point_1_j[j])])
        plt.plot(center_x, center_y, '+')


        #plt.plot(average_J_x[j], average_J_z[j], '-*', label='Jacobian experiment' + str(j))
        plt.plot(np.convolve(np.array(average_J_x[j]), kern, mode='valid')/4,
                 np.convolve(np.array(average_J_z[j]), kern, mode='valid')/4,
                 '-*', label='Jacobian experiment ' + str(j))
        plt.plot(average_J_x[j][0], average_J_z[j][0], 'rX')


    plt.legend(loc='best')
    plt.subplot(122)
    for j in range(len(average_P_x)):
        #plt.plot(np.mean(calibration_data_x_point_1_p[j]), np.mean(calibration_data_z_point_1_p[j]), 'ro')
        #plt.plot(np.mean(calibration_data_x_point_2_p[j]), np.mean(calibration_data_z_point_2_p[j]), 'bo')
        #plt.plot(np.mean(calibration_data_x_point_3_p[j]), np.mean(calibration_data_z_point_3_p[j]), 'go')
        #plt.plot(np.mean(calibration_data_x_point_4_p[j]), np.mean(calibration_data_z_point_4_p[j]), 'yo')
        center_x = (np.mean(calibration_data_x_point_4_p[j])/2 - np.mean(calibration_data_x_point_2_p[j])/2) + \
                   np.max([np.mean(calibration_data_x_point_4_p[j]), np.mean(calibration_data_x_point_2_p[j])]) + 5
        center_y = (np.mean(calibration_data_z_point_2_p[j])/2 - np.mean(calibration_data_z_point_1_p[j])/2) + \
                   np.min([np.mean(calibration_data_z_point_2_p[j]), np.mean(calibration_data_z_point_1_p[j])])
        plt.plot(center_x, center_y, '+')
        #plt.plot(average_P_x[j], average_P_z[j], '-*', label='P. Field experiment' + str(j))
        plt.plot(np.convolve(np.array(average_P_x[j]), kern, mode='valid')/4,
                 np.convolve(np.array(average_P_z[j]), kern, mode='valid')/4,
                 '-*', label='P. Field experiment ' + str(j))
        plt.plot(average_P_x[j][0], average_P_z[j][0], 'rX')

    plt.legend(loc='best')


def analyze_navigation_task(dir_folder):

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
        list_sensor_1_y_potential .append(data['sensor 1 y'].tolist())
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
            average_x.append((list_sensor_1_x_potential[j][i] + list_sensor_2_x_potential[j][i] + list_sensor_3_x_potential[j][i])/3)
            average_y.append((list_sensor_1_y_potential[j][i] + list_sensor_2_y_potential[j][i] + list_sensor_3_y_potential[j][i])/3)
            average_z.append((list_sensor_1_z_potential[j][i] + list_sensor_2_z_potential[j][i] + list_sensor_3_z_potential[j][i])/3)

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
        list_sensor_1_y_jacobian .append(data['sensor 1 y'].tolist())
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
            average_jx.append((list_sensor_1_x_jacobian[j][i] + list_sensor_2_x_jacobian[j][i] + list_sensor_3_x_jacobian[j][i])/3)
            average_jy.append((list_sensor_1_y_jacobian[j][i] + list_sensor_2_y_jacobian[j][i] + list_sensor_3_y_jacobian[j][i])/3)
            average_jz.append((list_sensor_1_z_jacobian[j][i] + list_sensor_2_z_jacobian[j][i] + list_sensor_3_z_jacobian[j][i])/3)

        average_J_x.append(average_jx)
        average_J_y.append(average_jy)
        average_J_z.append(average_jz)

    plt.figure()
    ax = plt.axes(projection='3d')
    for j in range(len(calibration_data_x_point_1_j)):
        ax.scatter3D(calibration_data_x_point_1_j[j], calibration_data_y_point_1_j[j],
                     calibration_data_z_point_1_j[j], cmap='Reds')
        ax.scatter3D(calibration_data_x_point_2_j[j], calibration_data_y_point_2_j[j],
                     calibration_data_z_point_2_j[j], cmap='Reds')
        ax.scatter3D(calibration_data_x_point_3_j[j], calibration_data_y_point_3_j[j],
                     calibration_data_z_point_3_j[j], cmap='Reds')
        ax.scatter3D(calibration_data_x_point_4_j[j], calibration_data_y_point_4_j[j],
                     calibration_data_z_point_4_j[j], cmap='Reds')
        ax.scatter3D(average_J_x[j], average_J_y[j], average_J_z[j])

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.figure()
    ax2 = plt.axes(projection='3d')
    for j in range(len(calibration_data_x_point_1_j)):
        ax2.scatter3D(calibration_data_x_point_1_p[j], calibration_data_y_point_1_p[j],
                     calibration_data_z_point_1_p[j], cmap='Reds')
        #ax.scatter3D(calibration_data_x_point_2_p[j], calibration_data_y_point_2_p[j],
        #             calibration_data_z_point_2_p[j], cmap='Reds')
        ax2.scatter3D(calibration_data_x_point_3_p[j], calibration_data_y_point_3_p[j],
                     calibration_data_z_point_3_p[j], cmap='Reds')
        ax2.scatter3D(calibration_data_x_point_4_p[j], calibration_data_y_point_4_p[j],
                     calibration_data_z_point_4_p[j], cmap='Reds')
        ax2.scatter3D(average_P_x[j], average_P_y[j], average_P_z[j])

    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    plt.show()

    plt.figure(3)
    for j in range(len(average_J_z)):
        center_x = (np.mean(calibration_data_x_point_4_j[j]) / 2 - np.mean(calibration_data_x_point_2_j[j]) / 2) + \
                   np.max([np.mean(calibration_data_x_point_4_j[j]), np.mean(calibration_data_x_point_2_j[j])]) + 5
        center_y = (np.mean(calibration_data_z_point_2_j[j]) / 2 - np.mean(calibration_data_z_point_1_j[j]) / 2) + \
                   np.min([np.mean(calibration_data_z_point_2_j[j]), np.mean(calibration_data_z_point_1_j[j])])
        print(center_x, center_y, 'jacobian')
        path_x = np.ones(len(average_J_y[j])) * center_x
        path_y = np.ones(len(average_J_y[j])) * center_y
        #plt.subplot(221)
        #plt.plot(average_J_y[j], average_J_z[j], '-*', label='Jacobian experiment' + str(j))
        #plt.subplot(222)
        #plt.plot(average_J_y[j], average_J_x[j], '-*', label='Jacobian experiment' + str(j))
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
        #plt.subplot(221)
        #plt.plot(average_P_y[j], average_P_z[j], '-*', label='P. Field experiment' + str(j))
        #plt.subplot(222)
        #plt.plot(average_P_y[j], average_P_x[j], '-*', label='P. Field experiment' + str(j))
        plt.subplot(223)
        plt.plot(average_P_y[j], np.absolute(average_P_z[j] - path_y), '-o')
        plt.subplot(224)
        plt.plot(average_P_y[j], np.absolute(average_P_x[j] - path_x), '-o')

    plt.figure(4)
    for j in range(len(list_target_x_jacobian)):
        center_point_x_j = np.ones(len(average_J_y[j])) * 150
        center_point_y_j = np.ones(len(average_J_y[j])) * 150

        plt.subplot(221)
        plt.plot(average_J_y[j], np.abs(list_target_x_jacobian[j]-center_point_x_j))
        plt.subplot(222)
        plt.plot(average_J_y[j], np.abs(list_target_y_jacobian[j] - center_point_y_j))

    for j in range(len(list_target_y_potential)):
        center_point_x_j = np.ones(len(average_P_y[j])) * 150
        center_point_y_j = np.ones(len(average_P_y[j])) * 150

        plt.subplot(223)
        plt.plot(average_P_y[j], np.abs(list_target_x_potential[j]-center_point_x_j))
        plt.subplot(224)
        plt.plot(average_P_y[j], np.abs(list_target_y_potential[j] - center_point_y_j))

    plt.subplot(221)
    plt.gca().set_title('Error X Axis')
    plt.ylabel('Jacobian')
    plt.subplot(222)
    plt.gca().set_title('Error Y Axis')
    plt.subplot(223)
    plt.ylabel('P. Field')


def analyze_centering_img(dir_folder):

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
        list_target_y_jacobian .append(data['filtered y'].tolist())

    for j, folder in enumerate(jacobian_results):
        j_son_file = [file for file in os.listdir(os.path.join(dir_folder, folder)) if file.endswith('.json')][0]

        data = pd.read_csv(os.path.join(dir_folder, folder, 'experiment_' + folder[-16:] + '_.csv'))
        list_target_x_potential.append(data['filtered x'].tolist())
        list_target_y_potential .append(data['filtered y'].tolist())

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


def extract_data(dir_folder, index_string_subfolder, atribute=''):

    list_folders = os.listdir(dir_folder)
    list_results = [folder for folder in list_folders if index_string_subfolder in folder]
    extracted_data = [[] for _ in range(len(list_results))]

    if atribute == 'center_points':
        for j, folder in enumerate(list_results):
            data = get_center_point_sensors(os.path.join(dir_folder, folder, 'experiment_' + folder[-16:] + '_.csv'))
            extracted_data[j].append(data)

    else:
        for j, folder in enumerate(list_results):
            print(folder)
            data = pd.read_csv(os.path.join(dir_folder, folder, 'experiment_' + folder[-16:] + '_.csv'))
            extracted_data[j].append(data[atribute].tolist())

    return extracted_data


def extract_data_0(dir_folder, atribute=''):

    list_files = os.listdir(dir_folder)
    file = [file for file in list_files if file.endswith('.csv')][0]
    print(file)

    if atribute == 'center_points':
        data = get_center_point_sensors(os.path.join(dir_folder, file))

    else:
        data = pd.read_csv(os.path.join(dir_folder, file))

    return data


def dimensionless_jerk(movement, fs):
    """
    Calculates the smoothness metric for the given speed profile using the dimensionless jerk
    metric.

    Parameters
    ----------
    movement : np.array
               The array containing the movement speed profile.
    fs       : float
               The sampling frequency of the data.
    Returns
    -------
    dl       : float
               The dimensionless jerk estimate of the given movement's smoothness.
    Notes
    -----

    Examples
    --------
    #>>> t = np.arange(-1, 1, 0.01)
    #>>> move = np.exp(-5*pow(t, 2))
    #>>> dl = dimensionless_jerk(move, fs=100.)
    #>>> '%.5f' % dl
    '-335.74684'
    """
    # first enforce data into an numpy array.
    movement = np.array(movement)

    # calculate the scale factor and jerk.
    movement_peak = np.amax(abs(movement))
    dt = 1. / fs
    movement_dur = len(movement) * dt
    jerk = np.diff(movement, 2) / pow(dt, 2)
    scale = pow(movement_dur, 3) / pow(movement_peak, 2)

    # estimate dj
    return - scale * sum(pow(jerk, 2)) * dt


def log_dimensionless_jerk(movement, fs):
    """
    Calculates the smoothness metric for the given speed profile using the log dimensionless jerk
    metric.

    Parameters
    ----------
    movement : np.array
               The array containing the movement speed profile.
    fs       : float
               The sampling frequency of the data.
    Returns
    -------
    ldl      : float
               The log dimensionless jerk estimate of the given movement's smoothness.
    Notes
    -----

    Examples
    --------
    #>>> t = np.arange(-1, 1, 0.01)
    #>>> move = np.exp(-5*pow(t, 2))
    #>>> ldl = log_dimensionless_jerk(move, fs=100.)
    #>>> '%.5f' % ldl
    '-5.81636'
    """
    return -np.log(abs(dimensionless_jerk(movement, fs)))


def spectral_arclength(movement, fs, padlevel=4, fc=10.0, amp_th=0.05):
    movement = check_nan(movement)
    """
    Calcualtes the smoothness of the given speed profile using the modified spectral
    arc length metric.
    Parameters
    ----------
    movement : np.array
               The array containing the movement speed profile.
    fs       : float
               The sampling frequency of the data.
    padlevel : integer, optional
               Indicates the amount of zero padding to be done to the movement
               data for estimating the spectral arc length. [default = 4]
    fc       : float, optional
               The max. cut off frequency for calculating the spectral arc
               length metric. [default = 10.]
    amp_th   : float, optional
               The amplitude threshold to used for determing the cut off
               frequency upto which the spectral arc length is to be estimated.
               [default = 0.05]
    Returns
    -------
    sal      : float
               The spectral arc length estimate of the given movement's
               smoothness.
    (f, Mf)  : tuple of two np.arrays
               This is the frequency(f) and the magntiude spectrum(Mf) of the
               given movement data. This spectral is from 0. to fs/2.
    (f_sel, Mf_sel) : tuple of two np.arrays
                      This is the portion of the spectrum that is selected for
                      calculating the spectral arc length.
    Notes
    -----
    This is the modfieid spectral arc length metric, which has been tested only
    for discrete movements.
    It is suitable for movements that are a few seconds long, but for long
    movements it might be slow and results might not make sense (like any other
    smoothness metric).
    Examples
    --------
    >>> t = np.arange(-1, 1, 0.01)
    >>> move = np.exp(-5*pow(t, 2))
    >>> sal, _, _ = spectral_arclength(move, fs=100.)
    >>> '%.5f' % sal
    '-1.41403'
    """
    # Number of zeros to be padded.
    nfft = int(pow(2, np.ceil(np.log2(len(movement))) + padlevel))

    # Frequency
    f = np.arange(0, fs, fs / nfft)
    # Normalized magnitude spectrum
    Mf = abs(np.fft.fft(movement, nfft))
    Mf = Mf / np.amax(Mf)

    # Indices to choose only the spectrum within the given cut off frequency Fc.
    # NOTE: This is a low pass filtering operation to get rid of high frequency
    # noise from affecting the next step (amplitude threshold based cut off for
    # arc length calculation).
    fc_inx = ((f <= fc) * 1).nonzero()
    f_sel = f[fc_inx]
    Mf_sel = Mf[fc_inx]

    # Choose the amplitude threshold based cut off frequency.
    # Index of the last point on the magnitude spectrum that is greater than
    # or equal to the amplitude threshold.
    inx = ((Mf_sel >= amp_th) * 1).nonzero()[0]
    fc_inx = range(inx[0], inx[-1] + 1)
    f_sel = f_sel[fc_inx]
    Mf_sel = Mf_sel[fc_inx]

    # Calculate arc length
    new_sal = -sum(np.sqrt(pow(np.diff(f_sel) / (f_sel[-1] - f_sel[0]), 2) + pow(np.diff(Mf_sel), 2)))
    return new_sal, (f, Mf), (f_sel, Mf_sel)


def get_center_point_sensors(dir_file):
    data = pd.read_csv(dir_file)

    points_x = []
    points_y = []
    points_z = []

    sensor_1_x = []
    sensor_1_y = []
    sensor_1_z = []

    sensor_2_x = []
    sensor_2_y = []
    sensor_2_z = []

    sensor_3_x = []
    sensor_3_y = []
    sensor_3_z = []

    sensor_1_x.append(data['sensor 1 x'].tolist())
    sensor_1_y.append(data['sensor 1 y'].tolist())
    sensor_1_z.append(data['sensor 1 z'].tolist())

    sensor_2_x.append(data['sensor 2 x'].tolist())
    sensor_2_y.append(data['sensor 2 y'].tolist())
    sensor_2_z.append(data['sensor 2 z'].tolist())

    sensor_3_x.append(data['sensor 3 x'].tolist())
    sensor_3_y.append(data['sensor 3 y'].tolist())
    sensor_3_z.append(data['sensor 3 z'].tolist())

    for j, list in enumerate(sensor_1_x):

        average_x = []
        average_z = []
        average_y = []

        for i in range(len(list)):
            average_x.append((sensor_1_x[j][i] + sensor_2_x[j][i] + sensor_3_x[j][i])/3)
            average_y.append((sensor_1_y[j][i] + sensor_2_y[j][i] + sensor_3_y[j][i])/3)
            average_z.append((sensor_1_z[j][i] + sensor_2_z[j][i] + sensor_3_z[j][i])/3)

        points_x.append(average_x)
        points_y.append(average_y)
        points_z.append(average_z)

    return points_x, points_y, points_z


def get_steps_length(list_time_stamps):
    list_length_steps = []
    for k in range(len(list_time_stamps) - 1):
        initial_time = datetime.strptime(list_time_stamps[k], '%Y-%m-%d %H:%M:%S.%f')
        end_time = datetime.strptime(list_time_stamps[k + 1], '%Y-%m-%d %H:%M:%S.%f')
        difference = (end_time - initial_time).total_seconds()
        list_length_steps.append(difference)

    return list_length_steps


def analyze_smoothness(dir_folder):
    param_to_analyze = 'time'
    time_stamps_jacobian = extract_data(dir_folder, "jacobian", param_to_analyze)
    time_stamps_potential_field = extract_data(dir_folder, "potential_field", param_to_analyze)

    param_to_analyze_2 = 'center_points'
    experiments_jacobian = extract_data(dir_folder, "jacobian", param_to_analyze_2)
    experiments_potential_field = extract_data(dir_folder, "potential_field", param_to_analyze_2)
    performances_jacobian = []
    performances_p_field = []
    num_peaks_jacobian = []
    num_peaks_p_field = []
    speed_metric_jacobian = []
    speed_metric_p_field = []

    plt.figure()
    for j, experiment_data in enumerate(experiments_jacobian):
        center_x = experiment_data[0][0][0]
        center_y = experiment_data[0][1][0]
        center_z = experiment_data[0][2][0]
        plt.subplot(221)
        plt.plot(center_y, center_x)
        plt.subplot(222)
        plt.plot(center_y, center_z)
        time_steps_jacobian = get_steps_length(time_stamps_jacobian[j][0])
        fs = 1./(np.median(time_steps_jacobian))
        performances_jacobian.append(spectral_arclength([center_x, center_y, center_z][2], fs)[0])
        speed_metric_jacobian.append(speed_metric(center_z))
        num_peaks_jacobian.append(number_of_peaks(center_z))

    performances_jacobian = check_nan(performances_jacobian)

    for j, experiment_data in enumerate(experiments_potential_field):
        center_x = experiment_data[0][0][0]
        center_y = experiment_data[0][1][0]
        center_z = experiment_data[0][2][0]
        plt.subplot(223)
        plt.plot(center_y, center_x)
        plt.subplot(224)
        plt.plot(center_y, center_z, label='experiment ' + str(j))
        time_steps_p_field = get_steps_length(time_stamps_potential_field[j][0])
        fs = 1./(np.median(time_steps_p_field))
        performances_p_field.append(spectral_arclength([center_x, center_y, center_z][2], fs)[0])
        speed_metric_p_field.append(speed_metric(center_z))
        num_peaks_p_field.append(number_of_peaks(center_z))

        plt.legend(loc='best')

    performances_p_field = check_nan(performances_p_field)

    print('Jacobian:')
    print('Smoothness')
    print(np.mean(performances_jacobian), np.median(performances_jacobian), np.std(performances_jacobian))
    #print('Speed metric')
    #print(np.mean(speed_metric_jacobian), np.median(speed_metric_jacobian), np.std(speed_metric_jacobian))
    print('Number of peaks')
    print(np.mean(num_peaks_jacobian), np.median(num_peaks_jacobian), np.std(num_peaks_jacobian))
    print('Potential Field:')
    print('Smoothness')
    print(np.mean(performances_p_field), np.median(performances_p_field), np.std(performances_p_field))
    #print('Speed metric')
    #print(np.mean(speed_metric_p_field), np.median(speed_metric_p_field), np.std(speed_metric_p_field))
    print('Number of peaks')
    print(np.mean(num_peaks_p_field), np.median(num_peaks_p_field), np.std(num_peaks_p_field))
    # Smoothness
    print('Smoothness')
    calculate_kruskal_p_value(performances_jacobian, performances_p_field)
    df = pd.DataFrame(np.array([performances_jacobian, performances_p_field]).T,
                      columns=['Jacobian', 'Potential Field'])
    sns.catplot(kind="violin", data=df).set(ylabel='smoothness', title='Smoothness')
    # Speed metric
    #calculate_kruskal_p_value(speed_metric_jacobian, speed_metric_jacobian)
    #df1 = pd.DataFrame(np.array([speed_metric_jacobian, speed_metric_jacobian]).T,
    #                  columns=['Jacobian', 'Potential Field'])
    #sns.catplot(kind="violin", data=df1).set(title='Speed Metric')
    # Number of peaks
    print('NUM PEAKS')
    calculate_kruskal_p_value(num_peaks_jacobian, num_peaks_p_field)
    df2 = pd.DataFrame(np.array([num_peaks_jacobian, num_peaks_p_field]).T,
                      columns=['Jacobian', 'Potential Field'])
    sns.catplot(kind="violin", data=df2).set(title='Num. Peaks')


def speed_metric(y):
    dy = np.gradient(y)
    speed_metric = np.mean(dy)/np.amax(dy)
    return speed_metric


def number_of_peaks(signal, prominence=0.05, plot=False):
    peaks, prom = find_peaks(signal, prominence=prominence)
    if plot == True:
        prominences = peak_prominences(signal, peaks)[0]
        plt.figure()
        plt.plot(signal)
        plt.plot(peaks, prominences + np.mean(signal), "x")
        plt.show()

    return len(peaks)


def analyze_time(dir_folder):
    to_analyze = 'time'
    experiments_jacobian = extract_data(dir_folder, "jacobian", to_analyze)
    experiments_potential_field = extract_data(dir_folder, "potential_field", to_analyze)

    performances_jacobian = []
    performances_p_field = []

    for j, experiment_data in enumerate(experiments_jacobian):
        initial_time = datetime.strptime(experiment_data[0][0], '%Y-%m-%d %H:%M:%S.%f')
        end_time = datetime.strptime(experiment_data[0][-1], '%Y-%m-%d %H:%M:%S.%f')
        difference = (end_time - initial_time).total_seconds()
        performances_jacobian.append(difference)

    for j, experiment_data in enumerate(experiments_potential_field):
        initial_time = datetime.strptime(experiment_data[0][0], '%Y-%m-%d %H:%M:%S.%f')
        end_time = datetime.strptime(experiment_data[0][-1], '%Y-%m-%d %H:%M:%S.%f')
        difference = (end_time - initial_time).total_seconds()
        performances_p_field.append(difference)

    plt.figure()
    plt.subplot(121)
    plt.boxplot(performances_jacobian)
    plt.subplot(122)
    plt.boxplot(performances_p_field)

    print('Jacobian:')
    print(np.mean(performances_jacobian), np.median(performances_jacobian), np.std(performances_jacobian))

    print('Potential Field:')
    print(np.mean(performances_p_field), np.median(performances_p_field), np.std(performances_p_field))
    calculate_kruskal_p_value(performances_jacobian, performances_p_field)

    df = pd.DataFrame(np.array([performances_jacobian, performances_p_field]).T,
                      columns=['Jacobian', 'Potential Field'])

    sns.catplot(kind="violin", data=df).set(ylabel='completion time (s)')


def visualize_calibration_points(dir_folder):

    list_folders = os.listdir(dir_folder)
    jacobian_results = [folder for folder in list_folders if "jacobian" in folder]
    potential_field = [folder for folder in list_folders if "potential_field" in folder]

    calibration_points_p_field = []
    calibration_points_jacobian = []
    id_names_p_field = []
    id_names_jacobian = []

    for j, folder in enumerate(potential_field):
        j_son_file = [file for file in os.listdir(os.path.join(dir_folder, folder)) if file.endswith('.json')][0]
        calibration_points = dm.read_data_json(os.path.join(dir_folder, folder, j_son_file))
        calibration_points_p_field.append(calibration_points)
        id_names_p_field.append(folder[-16:])

    for j, folder in enumerate(jacobian_results):
        j_son_file = [file for file in os.listdir(os.path.join(dir_folder, folder)) if file.endswith('.json')][0]
        calibration_points = dm.read_data_json(os.path.join(dir_folder, folder, j_son_file))
        calibration_points_jacobian.append(calibration_points)
        id_names_jacobian.append(folder[-16:])

    fig1 = plt.figure()
    ax1 = fig1.add_subplot(121, projection='3d')
    for j in range(len(calibration_points_p_field)):
        ax1.scatter3D(calibration_points_jacobian[j]['limit_point_x'],
                      calibration_points_jacobian[j]['limit_point_y'],
                      calibration_points_jacobian[j]['limit_point_z'],
                      label=id_names_jacobian[j])

    plt.legend(loc='best')
    ax2 = fig1.add_subplot(122, projection='3d')
    for j in range(len(calibration_points_p_field)):
        ax2.scatter3D(calibration_points_p_field[j]['limit_point_x'],
                      calibration_points_p_field[j]['limit_point_y'],
                      calibration_points_p_field[j]['limit_point_z'],
                      label=id_names_p_field[j])
    plt.legend(loc='best')
    points_x, points_y, points_z = determine_vertex_and_edges(calibration_points_p_field + 
                                                              calibration_points_jacobian, 
                                                              plot=False)

    x_1 = np.mean(points_x[0] + points_x[1] + points_x[2] + points_x[3])
    x_2 = np.mean(points_x[4] + points_x[5] + points_x[6] + points_x[7])

    y_1 = np.mean(points_y[0] + points_y[2] + points_y[4] + points_y[6])
    y_2 = np.mean(points_y[1] + points_y[3] + points_y[5] + points_y[7])

    z_1 = np.mean(points_z[0] + points_z[1] + points_z[4] + points_z[5])
    z_2 = np.mean(points_z[2] + points_z[3] + points_z[6] + points_z[7])

    print('h', x_1 - x_2)
    print('d', y_1 - y_2)
    print('w', z_2 - z_1)


def build_trajectory(dir_folder, trajectory_type):

    list_folders = os.listdir(dir_folder)
    jacobian_results = [folder for folder in list_folders if "jacobian" in folder]
    potential_field = [folder for folder in list_folders if "potential_field" in folder]

    calibration_points_p_field = []
    calibration_points_jacobian = []
    id_names_p_field = []
    id_names_jacobian = []

    for j, folder in enumerate(potential_field):
        j_son_file = [file for file in os.listdir(os.path.join(dir_folder, folder)) if file.endswith('.json')][0]
        calibration_points = dm.read_data_json(os.path.join(dir_folder, folder, j_son_file))
        calibration_points_p_field.append(calibration_points)
        id_names_p_field.append(folder[-16:])

    for j, folder in enumerate(jacobian_results):
        j_son_file = [file for file in os.listdir(os.path.join(dir_folder, folder)) if file.endswith('.json')][0]
        calibration_points = dm.read_data_json(os.path.join(dir_folder, folder, j_son_file))
        calibration_points_jacobian.append(calibration_points)
        id_names_jacobian.append(folder[-16:])

    points_x, points_y, points_z = determine_vertex_and_edges(calibration_points_p_field +
                                                              calibration_points_jacobian,
                                                              plot=False)

    x1 = np.mean(points_x[0] + points_x[1] + points_x[2] + points_x[3])
    x2 = np.mean(points_x[4] + points_x[5] + points_x[6] + points_x[7])

    y1 = np.mean(points_y[0] + points_y[2] + points_y[4] + points_y[6])
    y2 = np.mean(points_y[1] + points_y[3] + points_y[5] + points_y[7])

    z1 = np.mean(points_z[0] + points_z[1] + points_z[4] + points_z[5])
    z2 = np.mean(points_z[2] + points_z[3] + points_z[6] + points_z[7])

    set_of_points = [(x1, y1, z1),
                     (x1, y2, z1),
                     (x1, y1, z2),
                     (x1, y2, z2),
                     (x2, y1, z1),
                     (x2, y2, z1),
                     (x2, y1, z2),
                     (x2, y2, z2)]

    x, y, z = trajectories.build_trajectory(set_of_points, trajectory_type)
    return x, y, z


def plot_single_2D_path(experiments_data, save_dir):
    index_point = 10
    x1 = experiments_data[0][0][index_point]
    y1 = experiments_data[1][0][index_point]
    z1 = experiments_data[2][0][index_point]

    x2 = x1 - 20
    y2 = y1 - 5
    z2 = z1 + 12

    x3 = x1 - 20
    y3 = y1 + 10
    z3 = z1 - 12

    set_of_points1 = [(x1, y1, z1),
                      (x1, y2, z1),
                      (x1, y1, z2),
                      (x1, y2, z2),
                      (x2, y1, z1),
                      (x2, y2, z1),
                      (x2, y1, z2),
                      (x2, y2, z2)]

    set_of_points2 = [(x1, y1, z1),
                      (x1, y3, z1),
                      (x1, y1, z3),
                      (x1, y3, z3),
                      (x3, y1, z1),
                      (x3, y3, z1),
                      (x3, y1, z3),
                      (x3, y3, z3)]

    limx1, limy1, limz1 = trajectories.build_trajectory(set_of_points1, 'curve_left')
    limx2, limy2, limz2 = trajectories.build_trajectory(set_of_points2, 'curve_left')


    fig = plt.figure(figsize=(9, 7), dpi=80)
    ax = fig.add_subplot(1, 1, 1)

    if not (os.path.isdir(save_dir)):
        os.mkdir(save_dir)

    for j, point in enumerate(experiments_data[1][0][:]):
        print(j / len(experiments_data[1][0]) * 100, '%')
        ax.plot(experiments_data[1][0][:j], experiments_data[2][0][:j], 'bo')
        ax.plot(limy1, limz1, color='darkorange', linewidth=7.0)
        ax.plot(limy2, limz2, color='darkorange', linewidth=7.0)
        ax.fill_between(limy1, limz1, y2=limz2, color='bisque')
        ax.set_yticklabels('')
        ax.title.set_text('Top View')

        #ax.plot(experiments_data[1][0][:j], experiments_data[0][0][:j], 'bo')
        #ax.plot(limy1, limx1, color='orange', linewidth=7.0)
        #ax.plot(limy2, limx1 + 15, color='orange', linewidth=7.0)
        #ax.fill_between(limy1, limx1, y2=limx1 + 15, color='bisque')
        #ax.title.set_text('Side View')

        #ax.set_xlim(np.min(experiments_data[1][0]) - 10, np.max(experiments_data[1][0]) + 10)
        #ax.set_ylim(np.min(experiments_data[0][0]) - 20, np.max(experiments_data[0][0]) + 20)

        plt.savefig(save_dir + str(j).zfill(4) + '.png', bbox_inches='tight')
        ax.cla()

    return


def plot_single_3D_path(experiments_data, save_dir):
    fig = plt.figure(figsize=(9, 7), dpi=80)
    ax = fig.add_subplot(1,1,1, projection='3d')

    if not (os.path.isdir(save_dir)):
        os.mkdir(save_dir)

    for j, point in enumerate(experiments_data[1][0][:]):
        print(j / len(experiments_data[1][0]) * 100, '%')
        ax.scatter3D(experiments_data[1][0][:j], experiments_data[2][0][:j], experiments_data[0][0][:j], color='blue')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_xlim(np.min(experiments_data[1][0]) - 10, np.max(experiments_data[1][0]) + 10)
        ax.set_ylim(np.min(experiments_data[2][0]) - 20, np.max(experiments_data[2][0]) + 20)
        ax.set_zlim(np.min(experiments_data[0][0]) - 10, np.max(experiments_data[0][0]) + 10)

        plt.savefig(save_dir + str(j).zfill(4) + '.png', bbox_inches='tight')
        ax.cla()

    return


def plot_grid_style(experiments_data, save_dir):
    fig = plt.figure(figsize=(9, 7), dpi=80)
    # ax = fig.add_subplot(1,1,1, projection='3d')

    ax1 = fig.add_subplot(1, 2, 1, projection='3d')  # top and bottom left
    ax2 = fig.add_subplot(2, 2, 2)  # top right
    ax3 = fig.add_subplot(2, 2, 4)  # bottom right

    if not (os.path.isdir(save_dir)):
        os.mkdir(save_dir)

    index_point = 10
    x1 = experiments_data[0][0][index_point]
    y1 = experiments_data[1][0][index_point]
    z1 = experiments_data[2][0][index_point]

    x2 = x1 - 20
    y2 = y1 - 5
    z2 = z1 + 12

    x3 = x1 - 20
    y3 = y1 + 10
    z3 = z1 - 12


    set_of_points1 = [(x1, y1, z1),
                     (x1, y2, z1),
                     (x1, y1, z2),
                     (x1, y2, z2),
                     (x2, y1, z1),
                     (x2, y2, z1),
                     (x2, y1, z2),
                     (x2, y2, z2)]

    set_of_points2 = [(x1, y1, z1),
                      (x1, y3, z1),
                      (x1, y1, z3),
                      (x1, y3, z3),
                      (x3, y1, z1),
                      (x3, y3, z1),
                      (x3, y1, z3),
                      (x3, y3, z3)]

    limx1, limy1, limz1 = trajectories.build_trajectory(set_of_points1, 's_curve')
    limx2, limy2, limz2 = trajectories.build_trajectory(set_of_points2, 's_curve')
    test='test'
    y = [102.283333333, 102.383333333, 102.366666667, 102.37, 102.31, 102.356666667, 102.323333333, 102.363333333, 102.386666667, 102.433333333, 102.39, 102.363333333, 102.37, 102.366666667, 102.3, 102.296666667, 102.373333333, 102.39, 102.4, 102.36, 102.33, 102.33, 102.296666667, 102.366666667, 102.363333333, 102.37, 102.373333333, 102.346666667, 102.433333333, 102.356666667, 102.313333333, 102.336666667, 102.326666667, 102.356666667, 102.386666667, 102.393333333, 102.366666667, 102.35, 102.286666667, 102.346666667, 102.336666667, 102.393333333, 102.356666667, 102.38, 102.35, 102.423333333, 102.38, 102.37, 102.316666667, 102.35, 102.4, 102.4, 102.326666667, 102.366666667, 102.32, 102.32, 102.383333333, 102.306666667, 102.386666667, 102.3, 102.32, 102.31, 102.393333333, 102.36, 102.37, 102.326666667, 102.276666667, 102.4, 102.29, 102.326666667, 102.39, 102.35, 102.353333333, 102.336666667, 102.26, 102.373333333, 102.363333333, 102.326666667, 102.333333333, 102.366666667, 102.4, 102.43, 102.303333333, 102.413333333, 102.423333333, 102.446666667, 102.376666667, 102.486666667, 102.486666667, 102.58, 102.573333333, 102.673333333, 102.74, 102.833333333, 102.786666667, 102.886666667, 102.763333333, 102.82, 102.796666667, 102.913333333, 102.86, 102.903333333, 102.9, 102.846666667, 102.976666667, 102.93, 102.916666667, 102.446666667, 101.856666667, 101.55, 100.72, 100.193333333, 99.76, 99.2, 98.73, 98.0333333333, 97.6133333333, 96.8466666667, 96.6933333333, 95.77, 95.2266666667, 94.79, 94.1633333333, 93.76, 93.02, 92.5533333333, 91.9, 91.48, 90.6866666667, 90.3166666667, 89.87, 89.2866666667, 88.49, 88.05, 87.1266666667, 86.8566666667, 86.92, 86.82, 86.8233333333, 86.8033333333, 86.8966666667, 86.9533333333, 86.94, 86.9066666667, 86.9066666667, 86.8533333333, 86.9266666667, 86.91, 86.8733333333, 86.88, 86.93, 86.8766666667, 86.8766666667, 86.8966666667, 86.8766666667, 86.83, 86.91, 86.9433333333, 86.9866666667, 86.9266666667, 86.86, 86.9066666667, 86.85, 86.8666666667, 86.7966666667, 86.9166666667, 86.89, 86.9466666667, 86.8766666667, 86.87, 86.89, 86.8833333333, 86.8966666667, 86.7866666667, 86.8933333333, 86.9066666667, 86.97, 86.9333333333, 86.8833333333, 86.8966666667, 86.8466666667, 86.89, 86.89, 86.9533333333, 86.9033333333, 86.8733333333, 86.94, 86.88, 86.7866666667, 86.7566666667, 86.91, 86.87, 86.9166666667, 86.9066666667, 86.8033333333, 86.9666666667, 86.8933333333, 86.88, 86.7633333333, 86.9366666667, 86.8733333333, 86.8733333333, 86.94, 86.8933333333, 86.89, 86.88, 86.9066666667, 86.8533333333, 86.87, 86.88, 86.8933333333, 86.8933333333, 86.9166666667, 86.8866666667, 86.89, 86.8533333333, 86.94, 86.8966666667, 86.8433333333, 86.9666666667, 86.9233333333, 86.9033333333, 86.9266666667, 86.8633333333, 86.87, 86.8733333333, 86.8366666667, 86.8466666667, 86.9, 86.9033333333, 86.8366666667, 86.87, 86.8133333333, 86.8833333333, 86.8566666667, 86.8933333333, 86.8433333333, 86.8933333333, 86.8033333333, 86.7466666667, 86.3633333333, 85.8633333333, 84.6966666667, 84.4333333333, 84.5166666667, 84.42, 84.4766666667, 84.45, 84.5733333333, 84.5033333333, 84.5933333333, 85.0666666667, 85.0733333333, 84.6966666667, 84.83, 84.85, 84.8866666667, 84.82, 84.8333333333, 84.8266666667, 84.7466666667, 84.8566666667, 84.85, 84.84, 84.84, 84.83, 84.82, 84.8133333333, 84.8166666667, 84.85, 84.8233333333, 84.8466666667, 84.82, 84.9533333333, 84.8366666667, 84.8866666667, 84.8866666667, 84.7966666667, 84.8466666667, 84.8133333333, 84.8866666667, 84.9033333333, 84.8366666667, 84.8433333333, 84.8566666667, 84.8566666667, 84.89, 84.69, 84.8833333333, 84.82, 84.84, 84.86, 84.8733333333, 84.7666666667, 84.81, 84.71, 84.83, 84.85, 84.93, 84.94, 84.6866666667, 84.97, 84.79, 84.7, 84.9133333333, 84.8933333333, 84.8766666667, 84.8133333333, 84.7866666667, 84.6833333333, 84.7, 84.8, 84.6066666667, 84.8366666667, 84.6833333333, 84.8433333333, 84.77, 84.74, 84.7066666667, 84.2866666667, 83.62, 82.89, 82.37, 81.99, 81.2833333333, 80.5366666667, 80.1033333333, 79.6466666667, 79.0066666667, 78.5866666667, 78.1866666667, 77.4766666667, 76.9933333333, 76.2833333333, 75.81, 75.0733333333, 74.6666666667, 74.0266666667, 73.6, 73.14, 72.7066666667, 71.9366666667, 71.5333333333, 70.47, 70.2, 70.1866666667, 70.1766666667, 70.1766666667, 70.2533333333, 70.15, 70.2833333333, 70.2566666667, 70.1766666667, 70.3766666667, 70.3866666667, 70.5633333333, 70.62, 70.6166666667, 70.54, 70.6566666667, 70.5166666667, 70.5966666667, 70.6166666667, 70.5466666667, 70.4466666667, 70.5566666667, 70.5533333333, 70.64, 70.55, 70.6, 70.61, 70.5433333333, 70.6, 70.5166666667, 70.5466666667, 70.53, 70.5433333333, 70.5333333333, 70.53, 70.54, 70.5233333333, 70.47, 70.4866666667, 70.5333333333, 70.4133333333, 70.44, 70.47, 70.47, 70.67, 70.52, 70.6033333333, 70.4466666667, 70.51, 70.5033333333, 70.5366666667, 70.5033333333, 70.5133333333, 70.5866666667, 70.4966666667, 70.4833333333, 70.4866666667, 70.4933333333, 70.5, 70.51, 70.5033333333, 70.5166666667, 70.5033333333, 70.5033333333, 70.49, 70.4366666667, 70.4933333333, 70.47, 70.4633333333, 70.5433333333, 70.4966666667, 70.5266666667, 70.5266666667, 70.4633333333, 70.51, 70.4966666667, 70.5, 70.4733333333, 70.5033333333, 70.5033333333, 70.5233333333, 70.48, 70.4733333333, 70.49, 70.5033333333, 70.4333333333, 70.51, 70.5133333333, 70.53, 70.5366666667, 70.54, 70.48, 70.4933333333, 70.52, 70.5066666667, 70.51, 70.4966666667, 70.5466666667, 70.5166666667, 70.54, 70.5066666667, 70.4666666667, 70.4833333333, 70.5033333333, 70.5, 70.53, 70.5166666667, 70.51, 70.5, 70.52, 70.52, 70.4833333333, 70.4933333333, 70.5366666667, 70.52, 70.5066666667, 70.5233333333, 70.5033333333, 70.5133333333, 70.47, 70.47, 70.53, 70.5233333333, 70.5166666667, 70.5433333333, 70.5266666667, 70.4733333333, 70.5233333333, 70.4933333333, 70.5433333333, 70.5166666667, 70.5133333333, 70.5566666667, 70.5366666667, 70.5633333333, 70.4966666667, 70.56, 70.5733333333, 70.5266666667, 70.5633333333, 70.5933333333, 70.6, 70.6533333333, 70.56, 70.62, 70.6133333333, 70.5866666667, 70.5266666667, 70.2133333333, 69.13, 68.5566666667, 68.15, 67.67, 67.2266666667, 66.4866666667, 66.1033333333, 65.4733333333, 65.0766666667, 64.3633333333, 63.8966666667, 63.33, 62.8733333333, 62.1933333333, 61.75, 61.1566666667, 60.6366666667, 59.9566666667, 59.4433333333, 58.8466666667, 58.2833333333, 57.7333333333, 57.22, 56.19, 55.93, 55.9233333333, 55.93, 55.9033333333, 55.86, 55.8733333333, 55.8866666667, 55.8966666667, 56.2233333333, 56.1, 55.9133333333, 55.9933333333, 56.0666666667, 56.0533333333, 55.9033333333, 56.0333333333, 56.06, 56.0633333333, 56.06, 56.0633333333, 56.0766666667, 56.0333333333, 56.0566666667, 56.0333333333, 56.0733333333, 56.04, 56.0566666667, 56.08, 56.04, 56.1066666667, 56.1, 56.06, 56.07, 56.0766666667, 56.0366666667, 56.07, 56.0466666667, 56.08, 56.0833333333, 56.0766666667, 56.04, 56.0366666667, 56.0333333333, 56.0666666667, 56.0466666667, 56.0433333333, 56.0566666667, 56.0733333333, 56.05, 56.0566666667, 56.04, 56.0666666667, 56.0633333333, 56.0466666667, 56.0666666667, 56.07, 56.05, 56.0866666667, 56.04, 56.0433333333, 56.0266666667, 56.05, 56.03, 56.0566666667, 56.07, 56.0466666667, 56.0233333333, 56.06, 56.0566666667, 56.05, 56.0433333333, 56.0633333333, 56.0533333333, 56.1, 56.0666666667, 56.0666666667, 56.0666666667, 56.0433333333, 56.0633333333, 56.0566666667, 56.05, 56.0566666667, 56.06, 56.07, 56.0266666667, 56.04, 56.0433333333, 56.06, 56.1, 56.0433333333, 56.07, 56.0533333333, 56.0633333333, 56.0633333333, 56.04, 56.05, 56.0733333333, 56.0666666667, 56.0633333333, 56.07, 56.0566666667, 56.0466666667, 56.0966666667, 56.0666666667, 56.0466666667, 56.04, 56.0633333333, 56.0433333333, 56.0666666667, 56.0666666667, 56.1, 56.0033333333, 56.0533333333, 56.0533333333, 56.03, 56.05, 56.05, 56.0766666667, 56.0566666667, 56.0233333333, 56.05, 56.0466666667, 56.01, 56.0666666667, 56.0466666667, 56.0666666667, 56.0666666667, 56.0466666667, 56.0566666667, 56.0566666667, 56.0566666667, 56.0633333333, 56.0533333333, 56.0733333333, 56.08, 56.03, 56.04, 56.05, 56.0666666667, 56.0633333333, 56.06, 56.0633333333, 56.0566666667, 56.0933333333, 56.1, 56.1866666667, 56.1766666667, 56.1666666667, 56.18, 56.1866666667, 56.21, 56.22, 56.2233333333, 55.9833333333, 55.65, 55.42, 54.99, 54.3466666667, 53.9533333333, 53.4, 52.9033333333, 52.27, 51.7033333333, 51.14, 50.68, 50.0733333333, 49.6166666667, 49.0466666667, 48.51, 47.81, 47.3666666667, 46.81, 46.3766666667, 45.8433333333, 45.3033333333, 44.6666666667, 44.25, 43.58, 43.02, 42.2866666667, 41.9333333333, 41.3833333333, 40.82, 40.1366666667, 39.6666666667, 39.12, 38.5833333333, 37.95, 37.4233333333, 36.8866666667, 36.4, 35.87, 35.3866666667, 34.86, 34.36, 33.59, 33.17, 32.6266666667, 32.0466666667, 31.3333333333, 30.9266666667, 30.26, 29.72, 29.0833333333, 28.49, 27.8866666667, 27.37, 26.82, 26.4066666667, 25.8133333333, 25.4, 24.2566666667, 24.0766666667, 24.02, 23.7866666667, 23.6466666667, 23.6833333333, 23.6633333333, 23.6133333333, 23.7866666667, 23.6633333333, 23.7, 23.7066666667, 23.7133333333, 23.7266666667, 23.7033333333, 23.6866666667, 23.7133333333, 23.7066666667, 23.68, 23.68, 23.7, 23.7233333333, 23.6933333333, 23.6933333333, 23.7033333333, 23.7133333333, 23.6833333333, 23.7133333333, 23.68, 23.7166666667, 23.69, 23.6966666667, 23.6366666667, 23.7266666667, 23.7066666667, 23.7233333333, 23.6833333333, 23.7033333333, 23.7066666667, 23.67, 23.73, 23.6566666667, 23.67, 23.7, 23.7066666667, 23.6666666667, 23.71, 23.6866666667, 23.68, 23.7066666667, 23.69, 23.7033333333, 23.7133333333, 23.7433333333, 23.6933333333, 23.6866666667, 23.69, 23.7133333333, 23.7033333333, 23.7133333333, 23.6766666667, 23.6933333333, 23.69, 23.69, 23.7, 23.6666666667, 23.6766666667, 23.6866666667, 23.7066666667, 23.7066666667, 23.7166666667, 23.6833333333, 23.7, 23.71, 23.6933333333, 23.6833333333, 23.6866666667, 23.7066666667, 23.71, 23.6933333333, 23.6833333333, 23.72, 23.72, 23.6533333333, 23.73, 23.6833333333, 23.6966666667, 23.7066666667, 23.7133333333, 23.72, 23.6533333333, 23.6866666667, 23.6933333333, 23.6966666667, 23.7133333333, 23.7033333333, 23.7, 23.7366666667, 23.7266666667, 23.6833333333, 23.7066666667, 23.6633333333, 23.68, 23.71, 23.72, 23.6866666667, 23.7333333333, 23.7166666667, 23.63, 23.6633333333, 23.67, 23.6733333333, 23.7166666667, 23.68, 23.7466666667, 23.7233333333, 23.6566666667, 23.7133333333, 23.7033333333, 23.7166666667, 23.6833333333, 23.6933333333, 23.6566666667, 23.68, 23.7033333333, 23.6666666667, 23.6966666667, 23.7133333333, 23.66, 23.6766666667, 23.7566666667, 23.6666666667, 23.7066666667, 23.7066666667, 23.6766666667, 23.6866666667, 23.7133333333, 23.7066666667, 23.69, 23.7566666667, 23.6766666667, 23.6333333333, 23.7266666667, 23.7, 23.6833333333, 23.6833333333, 23.7, 23.7266666667, 23.6466666667, 23.69, 23.6333333333, 23.67, 23.6866666667, 23.6633333333, 23.6133333333, 23.6866666667, 23.6333333333, 23.6033333333, 23.7066666667, 23.6433333333, 23.6066666667, 23.6566666667, 23.65, 23.6633333333, 23.64, 23.6333333333, 23.6166666667, 23.62, 23.65, 23.6366666667, 23.63, 23.6666666667, 23.5866666667, 23.6266666667, 23.63, 23.45, 23.01, 21.9766666667, 21.81, 21.7433333333, 21.3666666667, 21.3633333333, 21.38, 20.8233333333, 20.4066666667, 19.72, 19.3133333333, 18.5533333333, 18.1933333333, 17.45, 16.38, 16.1966666667, 16.1966666667, 15.7366666667, 15.54, 15.54, 15.5433333333, 15.5433333333, 15.7166666667, 15.62, 15.75, 15.7733333333, 15.73, 15.75, 15.7366666667, 15.7366666667, 15.7633333333, 15.7433333333, 15.7433333333, 15.7433333333, 15.7066666667, 15.7266666667, 15.73, 15.7433333333, 15.7333333333, 15.7333333333, 15.75, 15.7633333333, 15.7233333333, 15.7166666667, 15.7333333333, 15.7, 15.74, 15.7366666667, 15.7466666667, 15.75, 15.7266666667, 15.7166666667, 15.7066666667, 15.7333333333, 15.7233333333, 15.72, 15.7166666667, 15.7266666667, 15.7333333333, 15.72, 15.7166666667, 15.72, 15.7133333333, 15.7266666667, 15.7066666667, 15.7233333333, 15.75, 15.7266666667, 15.7133333333, 15.7, 15.7166666667, 15.7333333333, 15.71, 15.69, 15.73, 15.7, 15.6766666667, 15.7233333333, 15.6966666667, 15.69, 15.7, 15.6833333333, 15.71, 15.7166666667, 15.7233333333, 15.7166666667, 15.71, 15.7166666667, 15.73, 15.7033333333, 15.71, 15.7033333333, 15.71, 15.7166666667, 15.6866666667, 15.71, 15.7233333333, 15.72, 15.7166666667, 15.6966666667, 15.7033333333, 15.7, 15.7166666667, 15.7, 15.7066666667, 15.7233333333, 15.6966666667, 15.6633333333, 15.72, 15.7266666667, 15.7233333333, 15.72, 15.6866666667, 15.7166666667, 15.69, 15.6933333333, 15.7, 15.68, 15.6866666667, 15.72, 15.7, 15.68, 15.7033333333, 15.7066666667, 15.68, 15.68, 15.73, 15.7133333333, 15.7433333333, 15.7266666667, 15.74, 15.7066666667, 15.74, 15.7466666667, 15.7366666667, 15.74, 15.73, 15.7333333333, 15.6966666667, 15.7466666667, 15.7433333333, 15.7666666667, 15.7833333333, 15.7466666667, 15.77, 15.7833333333, 15.7633333333, 15.7266666667, 15.7666666667, 15.79, 15.7533333333, 15.77, 15.7866666667, 15.7433333333, 15.7633333333, 15.7566666667, 15.7366666667, 15.7733333333, 15.76, 15.7866666667, 15.7433333333, 15.7533333333, 15.8066666667, 15.77, 15.78, 15.7533333333, 15.7533333333, 15.78, 15.7666666667, 15.7733333333, 15.7466666667, 15.78, 15.7866666667, 15.78, 15.7533333333, 15.73, 15.7466666667, 15.7666666667, 15.7666666667, 15.7666666667, 15.7433333333, 15.7833333333, 15.76, 15.7733333333, 15.7633333333, 15.76, 15.7433333333, 15.7733333333, 15.7466666667, 15.75, 15.7666666667, 15.7533333333, 15.7433333333, 15.7666666667, 15.7733333333, 15.7633333333, 15.75, 15.7533333333, 15.7566666667, 15.7666666667, 15.7433333333, 15.77, 15.7466666667, 15.7533333333, 15.76, 15.7666666667, 15.7566666667, 15.77, 15.7466666667, 15.7833333333, 15.7466666667, 15.7266666667, 15.75, 15.7466666667, 15.7633333333, 15.76, 15.7433333333, 15.7566666667, 15.7533333333, 15.76, 15.7466666667, 15.7433333333, 15.76, 15.7633333333, 15.75, 15.7533333333, 15.7533333333, 15.7433333333, 15.7466666667, 15.7466666667, 15.74, 15.7533333333, 15.7766666667, 15.7433333333, 15.7633333333, 15.7633333333, 15.7366666667, 15.72, 15.7466666667, 15.7533333333, 15.7366666667, 15.7466666667, 15.7333333333, 15.7466666667, 15.7433333333, 15.7366666667, 15.7433333333, 15.74, 15.75, 15.7333333333, 15.7466666667, 15.7466666667, 15.7466666667, 15.76, 15.7433333333, 15.7166666667, 15.7366666667, 15.7533333333, 15.7433333333, 15.7433333333, 15.7766666667, 15.7366666667, 15.7, 15.7366666667, 15.74, 15.75, 15.7066666667, 15.7433333333, 15.7466666667, 15.7333333333, 15.73, 15.7433333333, 15.7066666667, 15.72, 15.6933333333, 15.72, 15.69, 15.69, 15.69, 15.69, 15.7066666667, 15.6866666667, 15.7033333333, 15.68, 15.3966666667, 14.9866666667, 14.34]
    x = [-102.5, -102.5, -102.5, -102.5, -102.5, -102.5, -102.5, -102.5, -102.5, -102.5, -102.5, -102.5, -102.5, -102.5, -102.5, -102.5, -102.5, -102.5, -102.5, -102.5, -102.5, -102.5, -102.5, -102.5, -102.5, -102.5, -102.5, -102.5, -102.5, -102.5, -102.5, -102.5, -102.5, -102.5, -102.5, -102.5, -102.5, -102.5, -102.5, -102.5, -102.5, -102.5, -102.5, -102.5, -102.5, -102.5, -102.5, -102.5, -102.5, -102.5, -102.5, -102.5, -102.5, -102.5, -102.5, -102.5, -102.5, -102.5, -102.5, -102.5, -102.5, -102.5, -102.5, -102.5, -102.5, -102.5, -102.5, -102.5, -102.5, -102.5, -102.5, -102.5, -102.5, -102.5, -102.5, -102.5, -102.5, -102.5, -102.5, -102.5, -102.5, -102.5, -102.5, -102.5, -102.5, -102.5, -102.5, -102.5, -102.5, -102.5, -102.5, -102.5, -102.5, -102.5, -102.5, -102.5, -102.5, -102.5, -102.5, -102.5, -102.5, -102.5, -102.5, -102.5, -102.5, -102.5, -102.5, -102.5, -102.5, -102.5, -102.49425889199733, -102.48466016870944, -102.47329322290048, -102.45396150380724, -102.43369336191455, -102.39685346957347, -102.37071617707238, -102.31537139862593, -102.30311688272518, -102.22094774867526, -102.16586507841444, -102.11797044516024, -102.04357358737848, -101.99214990231906, -101.89056406534627, -101.82166542578595, -101.71889478874121, -101.64892306711933, -101.50837219318026, -101.43905515209946, -101.35216733477196, -101.23338620882978, -101.06139001082539, -100.96152580971523, -100.74059971837134, -100.673068988627, -100.68902903469004, -100.66379544011893, -100.6646395062231, -100.659572060171, -100.68315754771498, -100.69739961735596, -100.69405381995657, -100.68567511758276, -100.68567511758275, -100.67222695486987, -100.69070477509464, -100.68651390141878, -100.67727610979699, -100.67895753620891, -100.69154234079039, -100.67811692456958, -100.67811692456958, -100.68315754771498, -100.67811692456958, -100.66632702861682, -100.68651390141878, -100.69489057373413, -100.70574990687967, -100.69070477509464, -100.67391081919799, -100.68567511758276, -100.67138471791779, -100.67559387081684, -100.65788128496047, -100.68819085996364, -100.68147815241548, -100.69572712455405, -100.67811692456958, -100.67643509188235, -100.68147815241548, -100.67979794472382, -100.68315754771498, -100.65534359703233, -100.68231795160986, -100.68567511758276, -100.70157729821418, -100.69237970349344, -100.67979794472382, -100.68315754771498, -100.67054227776194, -100.68147815241548, -100.68147815241548, -100.69739961735596, -100.68483613069264, -100.67727610979699, -100.69405381995655, -100.67895753620891, -100.65534359703233, -100.64771955020151, -100.68651390141878, -100.67643509188235, -100.68819085996364, -100.68567511758276, -100.65957206017099, -100.70074216785288, -100.68231795160986, -100.67895753620891, -100.64941520676618, -100.69321686321258, -100.67727610979699, -100.67727610979699, -100.69405381995655, -100.68231795160986, -100.68147815241548, -100.67895753620891, -100.68567511758276, -100.67222695486987, -100.67643509188235, -100.67895753620891, -100.68231795160986, -100.68231795160986, -100.68819085996364, -100.68063815012306, -100.68147815241548, -100.67222695486987, -100.69405381995657, -100.68315754771498, -100.66969963439345, -100.7007421678529, -100.68986700639745, -100.68483613069264, -100.69070477509464, -100.67475244659167, -100.67643509188235, -100.67727610979699, -100.66801373798337, -100.67054227776194, -100.68399694073965, -100.68483613069264, -100.66801373798337, -100.67643509188235, -100.66210669805174, -100.67979794472382, -100.673068988627, -100.68231795160986, -100.66969963439345, -100.68231795160986, -100.659572060171, -100.64517453944242, -100.54623291442397, -100.41311154556357, -100.08441764536266, -100.00669284720016, -100.03143102301172, -100.00272254424132, -100.01957309376425, -100.0116509935161, -100.04817798061802, -100.02748174182106, -100.05407418728544, -100.1914180985545, -100.19332245204427, -100.08441764536266, -100.12327307607578, -100.1290725242043, -100.13968530641213, -100.12037052987424, -100.1242401733468, -100.12230576974704, -100.09902767312039, -100.1310040015114, -100.1290725242043, -100.12617374075776, -100.12617374075776, -100.12327307607578, -100.12037052987424, -100.11843445364417, -100.11940259630913, -100.1290725242043, -100.12133825435004, -100.12810647206409, -100.12037052987424, -100.1589165312318, -100.12520706157062, -100.13968530641213, -100.13968530641213, -100.11359060345244, -100.1281064720641, -100.11843445364417, -100.13968530641213, -100.1445009423179, -100.12520706157062, -100.12714021091875, -100.1310040015114, -100.1310040015114, -100.14064885130978, -100.0824660812647, -100.13872155263527, -100.12037052987424, -100.12617374075776, -100.13196942669927, -100.13582903792474, -100.1048584949414, -100.1174661018688, -100.08831825986934, -100.12327307607578, -100.1290725242043, -100.15219510131422, -100.15507696638693, -100.0814899849673, -100.16371129228503, -100.11165159930759, -100.08539311318452, -100.14738781785539, -100.14161218733872, -100.13679341840206, -100.11843445364417, -100.11068178346405, -100.08051367915672, -100.08539311318452, -100.11455979177487, -100.05800079533302, -100.12520706157062, -100.08051367915672, -100.12714021091875, -100.1058295661125, -100.09708239138507, -100.08734342042723, -99.96283426896225, -99.75831904623118, -99.52459469386872, -99.3518038412151, -99.22218613558543, -98.97355329906401, -98.70178885747856, -98.54406842262988, -98.37785534898164, -98.14491439905127, -97.99204690065947, -97.846458806953, -97.58803994062399, -97.412120994062, -97.15370212773298, -96.98142288351364, -96.71329814427088, -96.56528358233597, -96.3323426324056, -96.17704866578535, -96.0096223580229, -95.8519019231742, -95.57164484278923, -95.42484351496853, -95.03782183253213, -94.93954986928026, -94.93469693282337, -94.93105723048072, -94.93105723048072, -94.9589616151078, -94.92135135756695, -94.96988072213577, -94.96017484922201, -94.93105723048072, -95.00385127733395, -95.00749097967662, -95.07179238773031, -95.09241736767206, -95.09120413355784, -95.06329974893076, -95.1057629429285, -95.05480711013122, -95.08392472887252, -95.09120413355784, -95.06572621715921, -95.0293291937326, -95.06936591950186, -95.06815268538764, -95.09969677235739, -95.06693945127343, -95.08513796298674, -95.0887776653294, -95.06451298304498, -95.08513796298674, -95.05480711013122, -95.06572621715921, -95.0596600465881, -95.06451298304498, -95.06087328070232, -95.0596600465881, -95.06329974893076, -95.05723357835967, -95.03782183253213, -95.04388800310323, -95.06087328070232, -95.01719685259037, -95.02690272550414, -95.03782183253213, -95.03782183253213, -95.11061587938536, -95.05602034424544, -95.08635119710095, -95.0293291937326, -95.05238064190277, -95.04995417367434, -95.06208651481654, -95.04995417367434, -95.053593876017, -95.08028502652985, -95.0475277054459, -95.04267476898902, -95.04388800310323, -95.04631447133168, -95.04874093956012, -95.05238064190277, -95.04995417367434, -95.05480711013122, -95.04995417367434, -95.04995417367434, -95.04510123721745, -95.02568949138993, -95.04631447133167, -95.03782183253213, -95.03539536430368, -95.06451298304498, -95.0475277054459, -95.05844681247387, -95.05844681247387, -95.0353953643037, -95.05238064190277, -95.0475277054459, -95.04874093956012, -95.03903506664635, -95.04995417367434, -95.04995417367434, -95.05723357835966, -95.0414615348748, -95.03903506664635, -95.04510123721745, -95.04995417367434, -95.02447625727571, -95.05238064190277, -95.053593876017, -95.0596600465881, -95.06208651481654, -95.06329974893076, -95.0414615348748, -95.04631447133167, -95.05602034424544, -95.05116740778855, -95.05238064190277, -95.0475277054459, -95.06572621715921, -95.05480711013122, -95.06329974893076, -95.05116740778855, -95.03660859841791, -95.04267476898902, -95.04995417367434, -95.04874093956012, -95.05966004658809, -95.05480711013122, -95.05238064190277, -95.04874093956012, -95.05602034424544, -95.05602034424544, -95.04267476898902, -95.04631447133167, -95.06208651481654, -95.05602034424544, -95.05116740778855, -95.05723357835966, -95.04995417367434, -95.05359387601699, -95.03782183253213, -95.03782183253212, -95.0596600465881, -95.05723357835966, -95.05480711013122, -95.06451298304498, -95.05844681247387, -95.03903506664635, -95.05723357835967, -95.04631447133167, -95.06451298304498, -95.05480711013122, -95.053593876017, -95.06936591950186, -95.06208651481654, -95.07179238773031, -95.04752770544589, -95.07057915361608, -95.07543209007297, -95.05844681247387, -95.07179238773031, -95.0827114947583, -95.08513796298674, -95.10454970881426, -95.07057915361608, -95.09241736767206, -95.08999089944362, -95.08028502652985, -95.05844681247387, -94.94440280573714, -94.55010171861542, -94.34142545096947, -94.19341088903454, -94.01870517658676, -93.85734503939541, -93.58800706603841, -93.44848514290305, -93.21918389531534, -93.07480903572308, -92.81517693527985, -92.64532415928896, -92.43907435987144, -92.27286128622322, -92.0253615269222, -91.86596967703163, -91.658767917277, -91.48288036079927, -91.26082943889128, -91.09911699358933, -90.91748401084415, -90.75219000997129, -90.59656002824241, -90.45639181610612, -90.1898139185889, -90.12559249461316, -90.12396197320399, -90.12559249461316, -90.11907525764204, -90.10851231703401, -90.11175881793328, -90.11500854942733, -90.1174479684507, -90.19813654757183, -90.16744405715657, -90.12151770636177, -90.14112276777075, -90.15919639334538, -90.15590299566719, -90.11907525764204, -90.15096897012539, -90.15754928972885, -90.15837274033846, -90.15754928972885, -90.15837274033846, -90.16166856683586, -90.15096897012539, -90.15672604150795, -90.15096897012539, -90.16084430658549, -90.15261283593404, -90.15672604150795, -90.16249302951789, -90.15261283593404, -90.16909601954615, -90.16744405715659, -90.15754928972885, -90.1600202487582, -90.16166856683586, -90.15179080186536, -90.1600202487582, -90.1542575110918, -90.16249302951789, -90.16331769464017, -90.16166856683586, -90.15261283593404, -90.15179080186536, -90.15096897012539, -90.15919639334538, -90.15425751109179, -90.15343507233999, -90.15672604150795, -90.16084430658549, -90.155080152198, -90.15672604150795, -90.15261283593404, -90.15919639334538, -90.15837274033846, -90.1542575110918, -90.15919639334538, -90.1600202487582, -90.155080152198, -90.16414256221131, -90.15261283593404, -90.15343507233999, -90.14932591359732, -90.155080152198, -90.15014734070556, -90.15672604150795, -90.1600202487582, -90.1542575110918, -90.14850468879212, -90.15754928972885, -90.15672604150795, -90.155080152198, -90.15343507233999, -90.15837274033846, -90.15590299566719, -90.16744405715659, -90.15919639334538, -90.15919639334538, -90.15919639334538, -90.15343507233999, -90.15837274033846, -90.15672604150795, -90.155080152198, -90.15672604150795, -90.15754928972885, -90.1600202487582, -90.14932591359732, -90.15261283593404, -90.15343507233999, -90.15754928972885, -90.16744405715659, -90.15343507233999, -90.1600202487582, -90.15590299566719, -90.15837274033846, -90.15837274033846, -90.15261283593404, -90.155080152198, -90.16084430658549, -90.15919639334538, -90.15837274033846, -90.1600202487582, -90.15672604150795, -90.15425751109179, -90.16661837970395, -90.15919639334538, -90.15425751109179, -90.15261283593404, -90.15837274033846, -90.15343507233999, -90.15919639334538, -90.15919639334538, -90.16744405715659, -90.14358158784555, -90.15590299566719, -90.15590299566719, -90.15014734070556, -90.155080152198, -90.155080152198, -90.16166856683586, -90.15672604150795, -90.14850468879212, -90.155080152198, -90.15425751109179, -90.14522181243049, -90.15919639334538, -90.1542575110918, -90.15919639334538, -90.15919639334538, -90.15425751109179, -90.15672604150795, -90.15672604150795, -90.15672604150795, -90.15837274033846, -90.15590299566719, -90.16084430658549, -90.16249302951789, -90.15014734070556, -90.15261283593404, -90.155080152198, -90.15919639334538, -90.15837274033846, -90.15754928972885, -90.15837274033846, -90.15672604150795, -90.16579290473459, -90.16744405715659, -90.18898277082374, -90.18649054379387, -90.18400014098023, -90.1873210834349, -90.18898277082374, -90.1948050627175, -90.19730337216146, -90.19813654757183, -90.13866576765435, -90.05780564420672, -90.00318665304852, -89.90363511927868, -89.76089600362931, -89.67726769955914, -89.56427316954685, -89.46745870364836, -89.3502930675404, -89.2514046267674, -89.15863324661566, -89.08695431700323, -88.99799945999547, -88.93520908386503, -88.86184230859308, -88.79783002207682, -88.72169039035946, -88.67776180551782, -88.62730487645354, -88.59164242094234, -88.55208589636229, -88.51690020581589, -88.48169271182256, -88.46232302840602, -88.4372596790927, -88.42206041576475, -88.41006603818113, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139, -88.40748735954139]

    limx1 = np.ones(len(y)) * limx1[-1]
    limy1 = y
    limy2 = y

    limz1 = [point + 8 for point in x]
    limz2 = [point - 8 for point in x]


    """plt.figure()

    ax1 = fig.add_subplot(1, 2, 1, projection='3d')  # top and bottom left
    ax2 = fig.add_subplot(2, 2, 2)  # top right
    ax3 = fig.add_subplot(2, 2, 4)  # bottom right

    ax1.scatter3D(limy1, limx1, limz1, color='red')
    ax1.scatter3D(experiments_data[1][0][:], experiments_data[2][0][:], experiments_data[0][0][:], color='blue')
    ax1.set_xlabel('Y')
    ax1.set_ylabel('X')
    ax1.set_zlabel('Z')

    ax2.plot(experiments_data[1][0][:], experiments_data[2][0][:], 'bo')
    ax2.plot(limy1, limz1, color='darkorange', linewidth=7.0)
    ax2.plot(limy2, limz2, color='darkorange', linewidth=7.0)
    ax2.fill_between(limy1, limz1, y2=limz2, color='bisque')

    ax3.plot(experiments_data[1][0][:], experiments_data[0][0][:], 'bo')
    ax3.plot(limy1, limx1, color='orange')
    ax3.plot(limy2, limx1+15, color='orange')

    plt.show()"""

    for j, point in enumerate(experiments_data[1][0][:]):
        print(j / len(experiments_data[1][0]) * 100, '%')
        ax1.scatter3D(experiments_data[1][0][:j], experiments_data[2][0][:j], experiments_data[0][0][:j], color='blue')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        ax1.set_xlim(np.min(experiments_data[1][0]) - 10, np.max(experiments_data[1][0]) + 10)
        ax1.set_ylim(np.min(experiments_data[2][0]) - 20, np.max(experiments_data[2][0]) + 20)
        ax1.set_zlim(np.min(experiments_data[0][0]) - 10, np.max(experiments_data[0][0]) + 10)

        ax2.plot(experiments_data[1][0][:j], experiments_data[2][0][:j], 'bo')
        ax2.plot(limy1, limz1, color='darkorange', linewidth=7.0)
        ax2.plot(limy2, limz2, color='darkorange', linewidth=7.0)
        ax2.fill_between(limy1, limz1, y2=limz2, color='bisque')
        ax2.set_yticklabels('')
        ax2.title.set_text('Top View')
        # ax2.set_ylabel('X')
        # ax2.set_xlabel('Y')
        ax2.set_xlim(np.min(experiments_data[1][0]) - 10, np.max(experiments_data[1][0]) + 10)
        ax2.set_ylim(np.min(experiments_data[2][0]) - 20, np.max(experiments_data[2][0]) + 20)

        ax3.plot(experiments_data[1][0][:j], experiments_data[0][0][:j], 'bo')
        ax3.plot(limy1, limx1, color='orange', linewidth=7.0)
        ax3.plot(limy2, limx1 + 15, color='orange', linewidth=7.0)
        ax3.fill_between(limy1, limx1, y2=limx1 + 15, color='bisque')
        ax3.title.set_text('Side View')
        # ax3.set_ylabel('Z')
        # ax3.set_xlabel('Y')
        ax3.set_xlim(np.min(experiments_data[1][0]) - 10, np.max(experiments_data[1][0]) + 10)
        ax3.set_ylim(np.min(experiments_data[0][0]) - 20, np.max(experiments_data[0][0]) + 20)

        plt.savefig(save_dir + str(j).zfill(4) + '.png', bbox_inches='tight')
        ax1.cla()
        ax2.cla()
        ax3.cla()

    return


def save_video(image_folder, parent_folder):

    endoscopic_video = [file for file in os.listdir(parent_folder) if file.endswith('.avi')][0]
    video_name = 'video_trajectory_' + endoscopic_video[:-4] + '.avi'
    images = sorted([img for img in os.listdir(image_folder) if img.endswith(".png")])
    print(len(images), 'frames found')
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape
    video_dir = ''.join([parent_folder, video_name])
    video = cv2.VideoWriter(video_dir, cv2.VideoWriter_fourcc(*'DIVX'), 20, (width, height))

    for image in images:
        frame = cv2.imread(os.path.join(image_folder, image))
        video.write(frame)

    cv2.destroyAllWindows()
    video.release()

    # delete temp folder and files
    shutil.rmtree(image_folder)


def make_trajectory_video(dir_path):
    experiments_data = extract_data_0(dir_path, 'center_points')
    temp_dir = dir_path + 'temp/'
    plot_single_3D_path(experiments_data, temp_dir)
    #plot_single_2D_path(experiments_data, temp_dir)
    save_video(temp_dir, dir_path)


def determine_correspondece_axis(base_signal, test_signal):
    correspondence_list = []
    missing_list = []
    for x in test_signal:
        if x in base_signal:
            correspondence_list.append(x)
        else:
            missing_list.append(x)

    return correspondence_list, missing_list


if __name__ == '__main__':
    # plot 3_D data
    #directory_2 = os.getcwd() + '/results/n_control/straight_line/'
    #analyze_results(directory)
    #directory_1 = os.getcwd() + '/data/calibration/gt_trajectories/straight_line/'
    #plot_3D_data(directory_1)

    directory = '/media/benoit/0803-003A/Backup_Desktop_nearlab/Jorge/current_work/' \
                'robot_vision/to_analyze/task_2/path_2/experiment_potential_field_14_07_2021_19_14/'
    make_trajectory_video(directory)

    #build_trakectory(directory)

    #visualize_calibration_points(directory)
    #analyze_smoothness(directory)
    #analyze_time(directory)
    #plt.show()
