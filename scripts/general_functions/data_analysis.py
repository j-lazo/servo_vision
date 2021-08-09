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


def determine_vertex_and_edges(array_points):
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'olive', 'cyan', 'pink']
    raw_x = check_nan(array_points["limit_point_x"])
    raw_y = check_nan(array_points["limit_point_y"])
    raw_z = check_nan(array_points["limit_point_z"])

    points_x = [[] for _ in range(8)]
    points_y = [[] for _ in range(8)]
    points_z = [[] for _ in range(8)]

    for j in range(6):
        points_x[j] = raw_x[j*6:(j*6)+6]
        points_y[j] = raw_y[j*6:(j*6)+6]

    plt.subplot(221)
    for j in range(len(points_x)):
        print(len(points_x[j]))
        plt.plot(points_x[j], points_y[j], marker='.', color=colors[j], label='Experiment' + str(j))
    #plt.plot(np.median(points_x), np.median(points_y), 'o')
    #plt.subplot(222)
    #plt.plot(points_x, points_z, '*')
    #plt.plot(np.median(points_x), np.median(points_z), 'o')
    #plt.subplot(223)
    #plt.plot(points_y, points_z, '*')
    #plt.plot(np.median(points_y), np.median(points_z), 'o')


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
        performances_jacobian.append(log_dimensionless_jerk([center_x, center_y, center_z][2], fs))
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
        performances_p_field.append(log_dimensionless_jerk([center_x, center_y, center_z][2], fs))
        speed_metric_p_field.append(speed_metric(center_z))
        num_peaks_p_field.append(number_of_peaks(center_z))

        plt.legend(loc='best')

    performances_p_field = check_nan(performances_p_field)

    print('Jacobian:')
    print('Smoothness')
    print(np.mean(performances_jacobian), np.median(performances_jacobian), np.std(performances_jacobian))
    print('Speed metric')
    print(np.mean(speed_metric_jacobian), np.median(speed_metric_jacobian), np.std(speed_metric_jacobian))
    print('Number of peaks')
    print(np.mean(num_peaks_jacobian), np.median(num_peaks_jacobian), np.std(num_peaks_jacobian))
    print('Potential Field:')
    print(np.mean(performances_jacobian), np.median(performances_jacobian), np.std(performances_jacobian))
    print('Smoothness')
    print(np.mean(performances_p_field), np.median(performances_p_field), np.std(performances_p_field))
    print('Speed metric')
    print(np.mean(speed_metric_p_field), np.median(speed_metric_p_field), np.std(speed_metric_p_field))
    print('Number of peaks')
    print(np.mean(num_peaks_p_field), np.median(num_peaks_p_field), np.std(num_peaks_p_field))
    # Smoothness
    calculate_kruskal_p_value(performances_jacobian, performances_p_field)
    df = pd.DataFrame(np.array([performances_jacobian, performances_p_field]).T,
                      columns=['Jacobian', 'Potential Field'])
    sns.catplot(kind="violin", data=df).set(ylabel='smoothness', title='Smoothness')
    # Speed metric
    calculate_kruskal_p_value(speed_metric_jacobian, speed_metric_jacobian)
    df1 = pd.DataFrame(np.array([speed_metric_jacobian, speed_metric_jacobian]).T,
                      columns=['Jacobian', 'Potential Field'])
    sns.catplot(kind="violin", data=df1).set(title='Speed Metric')
    # Number of peaks
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
        print(j)
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


if __name__ == '__main__':
    # plot 3_D data
    #directory_2 = os.getcwd() + '/results/n_control/straight_line/'
    #analyze_results(directory)
    #directory_1 = os.getcwd() + '/data/calibration/gt_trajectories/straight_line/'
    #plot_3D_data(directory_1)
    directory = os.getcwd() + '/to_analyze/task_2/path_1/'
    analyze_smoothness(directory)
    #analyze_time(directory)
    plt.show()
