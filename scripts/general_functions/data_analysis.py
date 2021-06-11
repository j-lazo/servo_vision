import os
from matplotlib import pyplot as plt
import pandas as pd


def build_ground_truth_paths(directory_files, plot=True):

    list_files = os.listdir(directory_files)
    list_all_x = []
    list_all_y = []
    list_all_z = []
    for file in list_files:
        data = pd.read_csv(directory_files + file, skipinitialspace=True)
        list_all_x.append(data['sensor x'].tolist())
        list_all_y.append(data['sensor y'].tolist())
        list_all_z.append(data['sensor z'].tolist())

    if plot is True:
        plt.figure()
        for j in range(len(list_all_z)):
            plt.plot(list_all_y[j], list_all_z[j], 'o-')
            plt.ylim([-300, 300])
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
    directory = '/home/nearlab/Jorge/current_work/robot_vision/data/calibration/gt_trajectories/right_s/'
    build_ground_truth_paths(directory)
    plt.show()
