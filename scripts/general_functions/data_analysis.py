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
from sklearn.metrics import auc
from shapely.geometry import Polygon
import time
from sklearn.neighbors import NearestNeighbors


def rotation_matrix(axis, theta):
    axis = axis/np.sqrt(np.dot(axis, axis))
    a = np.cos(theta/2.)
    b, c, d = -axis*np.sin(theta/2.)

    return np.array([[a*a+b*b-c*c-d*d, 2*(b*c-a*d), 2*(b*d+a*c)],
                     [2*(b*c+a*d), a*a+c*c-b*b-d*d, 2*(c*d-a*b)],
                     [2*(b*d-a*c), 2*(c*d+a*b), a*a+d*d-b*b-c*c]])


def test_best_fit():
    N = 10  # number of random points in the dataset
    num_tests = 100  # number of test iterations
    dim = 3  # number of dimensions of the points
    noise_sigma = .01  # standard deviation error to be added
    translation = .1  # max translation of the test set
    rotation = .1  # max rotation (radians) of the test set

    # Generate a random dataset
    A = np.random.rand(N, dim)
    total_time = 0

    for i in range(num_tests):

        B = np.copy(A)
        # Translate
        t = np.random.rand(dim)*translation
        B += t

        # Rotate
        R = rotation_matrix(np.random.rand(dim), np.random.rand()*rotation)
        B = np.dot(R, B.T).T

        # Add noise
        B += np.random.randn(N, dim) * noise_sigma

        # Find best fit transform
        start = time.time()
        T, R1, t1 = best_fit_transform(B, A)
        total_time += time.time() - start

        # Make C a homogeneous representation of B
        C = np.ones((N, 4))
        C[:,0:3] = B

        # Transform C
        C = np.dot(T, C.T).T

        assert np.allclose(C[:,0:3], A, atol=6*noise_sigma) # T should transform B (or C) to A
        assert np.allclose(-t1, t, atol=6*noise_sigma)      # t and t1 should be inverses
        assert np.allclose(R1.T, R, atol=6*noise_sigma)     # R and R1 should be inverses

    print('best fit time: {:.3}'.format(total_time/num_tests))

    return

def test_icp():

    # https://github.com/ClayFlannigan/icp

    # Constants
    N = 10  # number of random points in the dataset
    num_tests = 1  # 100 number of test iterations
    dim = 3  # number of dimensions of the points
    noise_sigma = .01  # standard deviation error to be added
    translation = .1  # max translation of the test set
    rotation = .1  # max rotation (radians) of the test set

    # Generate a random dataset
    A = np.random.rand(N, dim)
    total_time = 0

    for i in range(num_tests):

        B = np.copy(A)
        print(np.shape(B))
        # Translate
        t = np.random.rand(dim)*translation
        B += t

        # Rotate
        R = rotation_matrix(np.random.rand(dim), np.random.rand() * rotation)
        B = np.dot(R, B.T).T

        # Add noise
        B += np.random.randn(N, dim) * noise_sigma

        # Shuffle to disrupt correspondence
        np.random.shuffle(B)

        # Run ICP
        start = time.time()
        T, distances, iterations = icp(B, A, tolerance=0.000001)
        total_time += time.time() - start

        # Make C a homogeneous representation of B
        C = np.ones((N, 4))
        C[:,0:3] = np.copy(B)

        # Transform C
        C = np.dot(T, C.T).T

        assert np.mean(distances) < 6*noise_sigma                   # mean error should be small
        assert np.allclose(T[0:3,0:3].T, R, atol=6*noise_sigma)     # T and R should be inverses
        assert np.allclose(-T[0:3,3], t, atol=6*noise_sigma)        # T and t should be inverses

    fig = plt.figure()
    ax1 = fig.add_subplot(221, projection='3d')
    ax2 = fig.add_subplot(222, projection='3d')
    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224)

    axs = [point[0] for point in A]
    ays = [point[1] for point in A]
    azs = [point[2] for point in A]

    cxs = [point[0] for point in B]
    cys = [point[1] for point in B]
    czs = [point[2] for point in B]

    ax1.scatter3D(axs, ays, azs)
    ax2.scatter3D(cxs, cys, czs)

    ax3.plot(axs, ays, '*')
    ax3.plot(cxs, cys, 'o')

    ax4.plot(axs, azs, '*')
    ax4.plot(cxs, czs, 'o')

    print('icp time: {:.3}'.format(total_time/num_tests))


def best_fit_transform(A, B):
    '''
    Calculates the least-squares best-fit transform that maps corresponding points A to B in m spatial dimensions
    Input:
      A: Nxm numpy array of corresponding points
      B: Nxm numpy array of corresponding points
    Returns:
      T: (m+1)x(m+1) homogeneous transformation matrix that maps A on to B
      R: mxm rotation matrix
      t: mx1 translation vector
    '''

    assert A.shape == B.shape

    # get number of dimensions
    m = A.shape[1]

    # translate points to their centroids
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B

    # rotation matrix
    H = np.dot(AA.T, BB)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)

    # special reflection case
    if np.linalg.det(R) < 0:
       Vt[m-1,:] *= -1
       R = np.dot(Vt.T, U.T)

    # translation
    t = centroid_B.T - np.dot(R,centroid_A.T)

    # homogeneous transformation
    T = np.identity(m+1)
    T[:m, :m] = R
    T[:m, m] = t

    return T, R, t


def nearest_neighbor(src, dst):
    '''
    Find the nearest (Euclidean) neighbor in dst for each point in src
    Input:
        src: Nxm array of points
        dst: Nxm array of points
    Output:
        distances: Euclidean distances of the nearest neighbor
        indices: dst indices of the nearest neighbor
    '''

    assert src.shape == dst.shape

    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(dst)
    distances, indices = neigh.kneighbors(src, return_distance=True)
    return distances.ravel(), indices.ravel()


def icp(A, B, init_pose=None, max_iterations=20, tolerance=0.001):
    '''
    The Iterative Closest Point method: finds best-fit transform that maps points A on to points B
    Input:
        A: Nxm numpy array of source mD points
        B: Nxm numpy array of destination mD point
        init_pose: (m+1)x(m+1) homogeneous transformation
        max_iterations: exit algorithm after max_iterations
        tolerance: convergence criteria
    Output:
        T: final homogeneous transformation that maps A on to B
        distances: Euclidean distances (errors) of the nearest neighbor
        i: number of iterations to converge
    '''

    assert A.shape == B.shape

    # get number of dimensions
    m = A.shape[1]

    # make points homogeneous, copy them to maintain the originals
    src = np.ones((m+1, A.shape[0]))
    dst = np.ones((m+1, B.shape[0]))
    src[:m, :] = np.copy(A.T)
    dst[:m, :] = np.copy(B.T)

    # apply the initial pose estimation
    if init_pose is not None:
        src = np.dot(init_pose, src)

    prev_error = 0

    for i in range(max_iterations):
        # find the nearest neighbors between the current source and destination points
        distances, indices = nearest_neighbor(src[:m,:].T, dst[:m,:].T)

        # compute the transformation between the current source and nearest destination points
        T,_,_ = best_fit_transform(src[:m,:].T, dst[:m,indices].T)

        # update the current source
        src = np.dot(T, src)

        # check error
        mean_error = np.mean(distances)
        if np.abs(prev_error - mean_error) < tolerance:
            break
        prev_error = mean_error

    # calculate final transformation
    T,_,_ = best_fit_transform(A, src[:m,:].T)

    return T, distances, i


def caculate_AUC(curve_1, curve_2):
    x_y_curve1 = [(0.121,0.232),(2.898,4.554),(7.865,9.987)] #these are your points for curve 1 (I just put some random numbers)
    x_y_curve2 = [(1.221,1.232),(3.898,5.554),(8.865,7.987)] #these are your points for curve 2 (I just put some random numbers)

    polygon_points = [] #creates a empty list where we will append the points to create the polygon

    for xyvalue in x_y_curve1:
        polygon_points.append([xyvalue[0], xyvalue[1]]) #append all xy points for curve 1

    for xyvalue in x_y_curve2[::-1]:
        polygon_points.append([xyvalue[0], xyvalue[1]]) #append all xy points for curve 2 in the reverse order (from last point to first point)

    for xyvalue in x_y_curve1[0:1]:
        polygon_points.append([xyvalue[0], xyvalue[1]]) #append the first point in curve 1 again, to it "closes" the polygon

    polygon = Polygon(polygon_points)
    area = polygon.area
    print(area)
    return area


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
            average_x.append(get_center_point_triangle(sensor_1_x[j][i], sensor_2_x[j][i], sensor_3_x[j][i]))
            average_y.append(get_center_point_triangle(sensor_1_y[j][i], sensor_2_y[j][i], sensor_3_y[j][i]))
            average_z.append(get_center_point_triangle(sensor_1_z[j][i], sensor_2_z[j][i], sensor_3_z[j][i]))

        points_x.append(average_x)
        points_y.append(average_y)
        points_z.append(average_z)

    return points_x, points_y, points_z


def get_center_point_triangle(corner_1, corner_2, corner_3, decimal_precision=2):
    return round(((corner_1 + corner_2 + corner_3)/3), decimal_precision)


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


def determine_error_acrros_axis(measured_points, gt_trajectory):

    error_x = []
    error_y = []
    error_z = []

    points_x = measured_points[0]
    points_y = measured_points[1]
    points_z = measured_points[2]

    x = gt_trajectory[0]
    y = gt_trajectory[1]
    z = gt_trajectory[2]

    for j, point in enumerate(points_z):
        error_z.append(np.abs(point - z[list(y).index(points_y[j])]))
        error_x.append(np.abs(points_x[j] - x[list(y).index(points_y[j])]))

    return error_x, error_y, error_z


def compare_trajectories(dir_folder):
    path_id = dir_folder[-2]
    trajectory_names = ['straight_line', 'curve_right', 'curve_left', 's_curve']
    trajectory_type = trajectory_names[int(path_id)-1]
    x, y, z = build_trajectory(dir_folder, trajectory_type)

    param_to_analyze_2 = 'center_points'
    experiments_jacobian = extract_data(dir_folder, "jacobian", param_to_analyze_2)
    experiments_potential_field = extract_data(dir_folder, "potential_field", param_to_analyze_2)
    performances_jacobian = []
    performances_p_field = []

    fig = plt.figure()
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224)
    auc_jacob = []
    auc_p_field = []
    max_error_jacob = []
    max_error_p_field = []
    mean_error_jacob = []
    mean_error_p_field = []

    for j, experiment_data in enumerate(experiments_jacobian):
        center_x = check_nan(experiment_data[0][0][0])
        center_y = check_nan(experiment_data[0][1][0])
        center_z = check_nan(experiment_data[0][2][0])
        ax1.plot(smooth(center_y), smooth(center_z), marker='+', label='experiment' + str(j))
        error_x, error_y, error_z = determine_error_acrros_axis([center_x,
                                                                 center_y,
                                                                 center_z],
                                                                [x, y, z])
        auc_jacob.append(auc(np.arange(0, len(error_z), 1), error_z))
        max_error_jacob.append(np.amax(error_z))
        mean_error_jacob.append(np.mean(error_z))
        ax3.plot(error_z)

    for j, experiment_data in enumerate(experiments_potential_field):
        center_x = check_nan(experiment_data[0][0][0])
        center_y = check_nan(experiment_data[0][1][0])
        center_z = check_nan(experiment_data[0][2][0])
        ax2.plot(smooth(center_y), smooth(center_z), marker='+', label='experiment ' + str(j))
        #a, b = determine_correspondece_axis(y, center_y)
        error_x, error_y, error_z = determine_error_acrros_axis([center_x,
                                                                 center_y,
                                                                 center_z],
                                                                [x, y, z])
        auc_p_field.append(auc(np.arange(0, len(error_z), 1), error_z))
        max_error_p_field.append(np.amax(error_z))
        mean_error_p_field.append(np.mean(error_z))
        ax4.plot(error_z)

    ax1.plot(y, z, color='darkorange', marker='*')
    ax2.plot(y, z, color='darkorange', marker='*')
    ax2.set_xlabel('y')
    ax3.set_ylabel('Abs. error z')
    ax4.set_ylabel('Abs. error z')

    print(np.mean(auc_p_field), np.median(auc_p_field), np.std(auc_p_field))
    print(np.mean(max_error_p_field), np.median(max_error_p_field), np.std(max_error_p_field))
    print(np.mean(mean_error_p_field), np.median(mean_error_p_field), np.std(mean_error_p_field))

    """print('AUC')
    calculate_kruskal_p_value(auc_jacob, auc_p_field)
    print(auc_jacob, auc_p_field)
    df = pd.DataFrame(np.array([auc_jacob, auc_p_field]).T,
                      columns=['Jacobian', 'Potential Field'])
    sns.catplot(kind="violin", data=df).set(ylabel='AUC', title='AUC')

    print('Max Error')
    calculate_kruskal_p_value(max_error_jacob, max_error_p_field)
    df = pd.DataFrame(np.array([max_error_jacob, max_error_p_field]).T,
                      columns=['Jacobian', 'Potential Field'])
    sns.catplot(kind="violin", data=df).set(ylabel='Error (mm)', title='Max Error')

    print('Mean Error')
    calculate_kruskal_p_value(mean_error_jacob, mean_error_p_field)
    df = pd.DataFrame(np.array([mean_error_jacob, mean_error_p_field]).T,
                      columns=['Jacobian', 'Potential Field'])
    sns.catplot(kind="violin", data=df).set(ylabel='Error (mm)', title='Mean Error')"""

    #ax3.plot(x, y)
    #ax4.plot(y, z)
    #plt.legend(loc='best')


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

    directory = os.getcwd() + '/to_analyze/task_2/path_4/'
    compare_trajectories(directory)

    #build_trajectory(directory)
    #visualize_calibration_points(directory)
    #analyze_smoothness(directory)
    #analyze_time(directory)

    #test_best_fit()
    #test_icp()

    #plt.show()