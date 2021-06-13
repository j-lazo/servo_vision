import os
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import glob


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

        #print(data['sensor z'].tolist())

    min_y, max_y = np.amin(np.amin(list_all_z)), np.amax(np.amax(list_all_z))
    print(min_y, max_y)
    min_x, max_x = np.amin(np.amin(list_all_y)), np.amax(np.amax(list_all_y))
    print(min_x, max_x)
    line_x = list(range(int(min_x), int(max_x)))
    line_y = np.ones((len(line_x)))*(min_y + (max_y-min_y)*0.5)
    upper_lim_1 = [line + 5 for line in line_y]
    upper_lim_2 = [line + 10 for line in line_y]
    down_lim_1 = [line - 5 for line in line_y]
    down_lim_2 = [line - 10 for line in line_y]
    vertical_left_x = [line_x[0], line_x[0]]
    vertical_left_y = [down_lim_1[0], upper_lim_1[0]]

    vertical_right_x = [line_x[-1], line_x[-1]]
    vertical_right_y = [down_lim_1[0], upper_lim_1[0]]
    paint_area = (list(zip(line_x, upper_lim_1)))
    paint_area = paint_area + (list(zip(line_x, down_lim_1)))
    print(paint_area)
    if plot is True:
        plt.figure()
        for j in range(len(list_all_z)):
            plt.plot(list_all_y[j], list_all_z[j], 'o-')
            plt.plot(line_x, line_y, 'r--')
            plt.plot(line_x, down_lim_1, color='darkorange')
            plt.plot(line_x, upper_lim_1, color='darkorange')
            plt.plot(line_x, upper_lim_2, color='darkorange')
            plt.plot(line_x, down_lim_2, color='darkorange')
            plt.plot(vertical_left_x, vertical_left_y, '--')
            plt.plot(vertical_right_x, vertical_right_y, '--')
            plt.fill([(60, -160), (120, -130), (80, -150), (80, -140)])
            plt.ylim([min_y-30, max_y+30])
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

        print(data['sensor z'].tolist())

    min_y, max_y = np.amin(np.amin(list_all_z)), np.amax(np.amax(list_all_z))
    print(min_y, max_y)
    if plot is True:
        plt.figure()
        for j in range(len(list_all_z)):
            plt.subplot(2,1,1)
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


if __name__ == '__main__':
    #directory = os.getcwd() + '/results/n_control/right_curve/'
    #analyze_results(directory)
    directory = os.getcwd() + '/data/calibration/gt_trajectories/straight_line/'
    build_ground_truth_paths(directory)
    plt.show()
