from __future__ import division
from __future__ import print_function

import warnings
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import math
import pytweening
import glob
import os
import re
import time
from tqdm import *
from pathlib import Path
from PIL import Image
from os import path
from datasaurus_dozen_implem import save_scatter_and_results
from scipy.spatial import distance_matrix
from scipy.optimize import linear_sum_assignment as hung
from evaluation import save_pdf

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=UserWarning)

# setting up the style for the charts
sns.set_style("darkgrid")
mpl.rcParams['font.size'] = 12.0
mpl.rcParams['text.color'] = '#222222'
mpl.rcParams['pdf.fonttype'] = 42
current_path = Path(__file__).resolve().parent
max_time = 1800


# from https://github.com/khuyentran1401/same-stats-different-graphs/tree/master
# This function calculates the summary statistics for the given set of points
def get_values(df):

    xm = df.x.mean()
    ym = df.y.mean()
    xsd = df.x.std()
    ysd = df.y.std()
    pc = df.corr().x.y

    return [xm, ym, xsd, ysd, pc]


# from https://github.com/khuyentran1401/same-stats-different-graphs/tree/master
# checks to see if the statistics are still within the acceptable bounds
# with df1 as the original dataset, and df2 as the one we are testing
def is_error_still_ok(df1, df2, decimals=2):
    r1 = get_values(df1)
    r2 = get_values(df2)

    # check each of the error values to check if they are the same to the correct number of decimals
    r1 = [math.floor(r * 10 ** decimals) for r in r1]
    r2 = [math.floor(r * 10 ** decimals) for r in r2]

    # we are good if r1 and r2 have the same numbers
    er = np.subtract(r1, r2)
    er = [abs(n) for n in er]

    return np.max(er) == 0


# less precise chamfer algorithm, allows for multiple points to overlap, good for first initialization
def chamfer(pos_1, pos_2):

    pdist = distance_matrix(pos_1, pos_2)
    term1 = pdist.min(axis=1).mean()
    term2 = pdist.min(axis=0).mean()

    return term1 + term2


def single_point_distance(point, pos_2):

    pdist = distance_matrix(point[None, :], pos_2)

    return pdist.min()


# precise hungarian algorithm, measures most optimal distances between every pair of points (O(n^3)), good for refining results
def hungarian(pos_1, pos_2):

    # square the distances so larger distances weigh more heavily
    pdist = distance_matrix(pos_1, pos_2) ** 2
    row_idcs, col_idcs = hung(pdist)

    # sqrt the distances for the final sum
    return np.sqrt(pdist)[row_idcs, col_idcs].sum()


# helper function to scale all data between 0 and 1, manually scale between 20, 80
def scale(x0):

    mtx3 = (x0 - np.min(x0)) / (np.max(x0) - np.min(x0))
    mtx3 *= 60
    mtx3 += 24

    return mtx3


# # # helper function to normalize data
# def normalize_shape(X):
#
#     X = X - X.mean(axis=0)
#     X = X / X.std(axis=0, ddof=0)
#
#     return X


# inspired by and adapted from https://github.com/khuyentran1401/same-stats-different-graphs/tree/master
# This is the function which does one round of perturbation
# df: is the current dataset
# tar_df: is the target dataset
# sample_size: how many points we move in one perturbation
# shake: the maximum amount of movement in each iteration
# temp: the temperature, how often are we accepting bad results
# x_bounds and y_bounds: boundaries of the scatterplot, set to 0 and 100
# dis_func: which distance function to use
# min_move: the minimum distance each move should be
def perturb(df, tar_df,
            shake=0.1,
            sample_size=20,
            temp=0,
            x_bounds=[0, 100], y_bounds=[0, 100],
            dis_func=chamfer, min_move = 5):

    # this is the simulated annealing step, if "do_bad", then we are willing to
    # accept a new state which is worse than the current one
    do_bad = np.random.random_sample() < temp

    scaled_tar_df = scale(tar_df)
    old_dist = dis_func(scaled_tar_df, scale(df.to_numpy()))

    while True:

        # take multiple rows at random and shift them
        row = np.random.randint(0, len(df), sample_size)

        # save old vals
        old_vals = [df['x'][row], df['y'][row]]

        # perturb the new rows
        i_xm = df['x'][row]
        i_ym = df['y'][row]
        xm = i_xm + np.random.randn() * shake
        ym = i_ym + np.random.randn() * shake

        # if our new dataset is out of bounds then we can skip the rest, redo the above
        # if not ((xm >= x_bounds[0]).all() & (xm <= x_bounds[1]).all() &
        #         (ym >= y_bounds[0]).all() & (ym <= y_bounds[1]).all()):
        #     continue

        # set new vals and compute the distance between current dataset and target dataset
        df['x'][row] = xm
        df['y'][row] = ym
        new_dist = dis_func(scaled_tar_df, scale(df.to_numpy()))

        # # we accept new vals if we are closer (with a minimum amount) or if we are allowed to accept bad solution
        if (new_dist < old_dist and (abs(new_dist - old_dist) >= min_move)) or do_bad:
            break
        # we accept new vals if we are closer (with a minimum amount) or if we are allowed to accept bad solution
        # if (new_dist < old_dist) or do_bad:
        #     break
        else:
            # set back to old vals if our solution is unacceptable
            df['x'][row] = old_vals[0]
            df['y'][row] = old_vals[1]

    return df, new_dist


# from: https://github.com/khuyentran1401/same-stats-different-graphs/tree/master
# helper function for setting the shake, temperature and sample size
def s_curve(v):
    return pytweening.easeInOutQuad(v)


# inspired by and adapted from: https://github.com/khuyentran1401/same-stats-different-graphs/tree/master
# this is the main function, for taking one dataset and perturbing it into a target shape
# df: the initial dataset
# target: the shape we are aiming for
# directory: where to save results
# iters: how many iterations to run the algorithm for
# num_frames: how many frames to save to disk (for animations)
# decimals: how many decimal points to keep fixed
# max_shake: the step size at the start
# min_shake: the step size near the end of the process, the step size changes fros max to min over time
# max_sample: the sample size at the start
# min_sample: the sample size near the end of the process, the sample size changes fros max to min over time
# max_temp: the temperature at the start
# min_temp: the temperature near the end of the process, the temperature changes fros max to min over time
# function_calls: list of what function to call at each iteration
#
def run_pattern(df, target, directory, iters=100000, num_frames=100, decimals=2, max_shake=0.6, min_shake=0.1,
                max_sample=20, min_sample=1,
                max_temp=0.4, min_temp=0,
                function_calls=None,
                ramp_in=False, ramp_out=False, freeze_for=0,
                labels=["X Mean", "Y Mean", "X SD", "Y SD", "Corr."],
                reset_counts=False):

    global frame_count
    global it_count

    if reset_counts:
        it_count = 0
        frame_count = 0

    # load target dataframe and scale
    r_good = df.copy()
    tar_df = pd.read_csv("target_datasets/{}.csv".format(target), header=None,
                         names=['x', 'y'])
    tar_df = tar_df.to_numpy()

    # this is a list of frames that we will end up writing to file
    write_frames = [int(round(pytweening.linear(x) * iters)) for x in np.arange(0, 1, 1 / (num_frames - freeze_for))]

    if ramp_in and not ramp_out:
        write_frames = [int(round(pytweening.easeInSine(x) * iters)) for x in
                        np.arange(0, 1, 1 / (num_frames - freeze_for))]
    elif ramp_out and not ramp_in:
        write_frames = [int(round(pytweening.easeOutSine(x) * iters)) for x in
                        np.arange(0, 1, 1 / (num_frames - freeze_for))]
    elif ramp_out and ramp_in:
        write_frames = [int(round(pytweening.easeInOutSine(x) * iters)) for x in
                        np.arange(0, 1, 1 / (num_frames - freeze_for))]

    extras = [iters] * freeze_for
    write_frames.extend(extras)

    looper = trange(iters + 1, leave=True, ascii=True, desc=target + " pattern")
    best_dis = 1e9

    func_list = function_calls
    prev_func = func_list[0]

    # get the distance between point clouds at the start and create the minimum movement needed based off that and the number of iterations
    start_tot_dis = prev_func(df.to_numpy(), scale(tar_df))
    min_move = start_tot_dis / (iters * 1.5)
    dis_progression = [0] * (iters + 1)

    start_time = time.time()
    # this is the main loop, were we run for many iterations to come up with the pattern
    for i in looper:

        # set the current temperature, shake, sample size and distance function depending on which iteration we are in
        t = (max_temp - min_temp) * s_curve(((iters - i) / iters)) + min_temp
        curr_shake = (max_shake - min_shake) * s_curve(((iters - i) / iters)) + min_shake
        curr_sample_size = int((max_sample - min_sample) * s_curve(((iters - i) / iters)) + min_sample)
        curr_func = func_list[i]

        # when we switch to a different distance function we need to reset the loss so we set it to an arbitrary large value
        if prev_func != curr_func:
            # curr_sample_size = 1
            best_dis = 1e9
            start_tot_dis = curr_func(df, scale(tar_df))
            min_move = start_tot_dis / (iters * 1.5)

        # main jittered result and new distance
        test_good, new_dis = perturb(r_good.copy(), temp=t, tar_df=tar_df,
                                     shake=curr_shake, sample_size=curr_sample_size, dis_func=curr_func, min_move=min_move)

        if i == 0:
            dis_progression[i] = new_dis

        # here we are checking that after the purturbation, that the statistics are still within the allowable bounds
        if is_error_still_ok(df, test_good, decimals):
            r_good = test_good

            # tracking of distance (loss) and adding it to the tqdm thing
            if new_dis < best_dis:
                best_dis = new_dis

            looper.set_description("Current loss: {} | Best loss: {}".format(str(round(new_dis, 4)), str(round(best_dis, 4))))

        dis_progression[i] = new_dis

        # save this chart to the file
        for x in range(write_frames.count(i)):
            save_scatter_and_results(r_good, target + "-image-" + format(int(frame_count), '05'), dp=150, labels=labels,
                                     directory=directory)
            r_good.to_csv(f'{directory}/{target}' + "-data-" + format(int(frame_count), '05') + ".csv", index=False, header=False)

            frame_count = frame_count + 1

        prev_func = func_list[i]

        if (time.time() - start_time) > max_time:
            break

    return r_good


# inspired by and adapted from: https://github.com/khuyentran1401/same-stats-different-graphs/tree/master
# function to load a dataset, and then perturb it
# start_dataset: name of the starting dataset
# target: the name of the target dataset
# iterations: how many iterations to run the algorithm for
# decimals: how many decimal points to keep fixed
# num_frames: how many frames to save to disk (for animations)
# max_temp: the temperature at the start
# min_temp: the temperature near the end of the process, the temperature changes fros max to min over time
# max_shake: the step size at the start
# min_shake: the step size near the end of the process, the step size changes fros max to min over time
# max_sample_divis: integer that determines how large our sample will be, divides the number of points in df by this integer
# min_sample: the sample size near the end of the process, the sample size changes fros max to min over time
# function_calls: list of what function to call at each iteration
def do_single_run(start_dataset, target, iterations=100000, decimals=2, num_frames=100, max_temp=0.4, min_temp=0,
                  max_shake=0.6, min_shake=0.1, max_sample_divis=35, min_sample=1, function_calls=None):

    global it_count
    global frame_count

    it_count = 0
    frame_count = 0

    # load dataset
    df = pd.read_csv("seed_datasets/{}.csv".format(start_dataset), index_col=0)

    # set the maximum sample size based on the sample division arg
    max_sample = int(len(df) / max_sample_divis)

    # if we don't have specified distance functions then set the chamfer distance to be the default
    if function_calls is None:
        function_calls = [chamfer] * (iterations + 1000)

    temp = run_pattern(df, target, iters=iterations, num_frames=num_frames, directory=f'datadance_doubledozen/results//{start_dataset}_{target}',
                       decimals=decimals, max_temp=max_temp, min_temp=min_temp, max_shake=max_shake,
                       min_shake=min_shake, max_sample=max_sample, min_sample=min_sample, function_calls=function_calls)
    return temp


# function to extract the numeric part from the filename for sorting
def extract_number(filename):
    match = re.findall(r'\d+', filename)  # Find the first number in the filename
    return int(match[-1]) if match else 0


# inspired by and adapted from: https://github.com/khuyentran1401/same-stats-different-graphs/tree/master
def create_gifs(shape_start, shape_end):

    # create the frames from all png files
    imgs = glob.glob(f"datadance_doubledozen/results/{shape_start}_{shape_end}/*.png")

    # get all the PNG files in the directory (sorted by the numeric part of the filename)
    frames = [Image.open(os.path.join(filename)) for filename in
              sorted(imgs, key=extract_number)]

    # Save into a GIF file that loops forever
    if not path.exists('datadance_doubledozen/results/gifs'):
        os.mkdir('datadance_doubledozen/results/gifs/')

    frames[0].save(f"datadance_doubledozen/results/gifs/{shape_start}_{shape_end}.gif", format='GIF',
                   append_images=frames[1:],
                   save_all=True,
                   duration=700 // 6, loop=0)


if __name__ == '__main__':

    # SET ALL ARGUMENTS HERE
    # for the exact functionality of the Datasaurus Dozen paper use the following parameter values
    # n_points = 250
    # it = 300000
    # de = 2
    # frames = 100
    # max_shake = 0.1
    # min_shake = 0.1
    # max_temp = 0.4
    # min_temp = 0
    # max_sample_divis = 250  # equivalant to a sample size of 1
    # min_sample = 1
    # func_list = [single_point_distance] * (int(it) + 1000)

    # SET ALL ARGUMENTS HERE
    it = 200000  # 100000 for 2nd round of datadance, 150000 normal
    de = 2
    frames = 100
    max_shake = 0.5  # 0.25 for 2nd round of datadance, 0.5 normal
    min_shake = 0.1  # 0.1 normal
    max_temp = 0.4  # 0.15 for 2nd round of datadance, 0.4 normal, 0.15 test
    min_temp = 0
    max_sample_divis = 35  # 40 for 2nd round of datadance, 35 normal
    min_sample = 1  # 1 normal

    # distance functions
    func_list = [chamfer] * int(it * 0.80) + [hungarian] * (int(it * 0.2) + 1000)  # for normal Datasaurus Dozen
    # func_list = [hungarian] * int(it * 0.85) + [chamfer] * (int(it * 0.15) + 1000)  # for datadance

    shape_ends = ['x_142', 'h_lines_142', 'v_lines_142', 'wide_lines_142', 'high_lines_142', 'slant_up_142', 'slant_down_142', 'circle_142', 'star_142', 'down_parab_142', 'bullseye_142', 'dots_142', 'datasaurus_142']

    t_xm = 54.265
    t_ym = 47.835
    # t_xsds = list(range(5, 65, 5))  # add 0.505
    t_xsd = 16.765
    t_ysd = 26.935
    t_pc = -0.065
    # t_pcs = np.arange(0, 1, 0.1)

    # for i in t_pcs:
    for shape_end in shape_ends:
        # t_pc = round(i + 0.055, 3)  # 0.505 for xsd, 0.055 for pc
        # t_xsd = i + 0.505
        shape_start = 'random_cloud_n142_xm{}_ym{}_xsd{}_ysd{}_pc{}'.format(str(t_xm).replace('.', '-'),
                                                                            str(t_ym).replace('.', '-'), str(t_xsd).replace('.', '-'),
                                                                            str(t_ysd).replace('.', '-'), str(t_pc).replace('.', '-'))
        # shape_end = 'star_142'

        print('Doing shape: ' + shape_end)
        save_directory = f'datadance_doubledozen/results/{shape_start}_{shape_end}'

        # check if we have the seed dataset and target dataset
        if (shape_start + '.csv' in os.listdir('seed_datasets')) and (shape_end + '.csv' in os.listdir('target_datasets')):

            # make the directory if we have to
            if path.exists(save_directory):
                print(f"File {save_directory} exists")
            else:
                os.mkdir(save_directory)

            temp = do_single_run(shape_start, shape_end, iterations=it, decimals=de, num_frames=frames, max_shake=max_shake,
                          min_shake=min_shake, max_temp=max_temp, min_temp=min_temp, max_sample_divis=max_sample_divis,
                          min_sample=min_sample, function_calls=func_list)

            save_pdf(temp[['x', 'y']], df_target=pd.read_csv("target_datasets/{}.csv".format(shape_end), header=None, names=['x', 'y']),
                     save_location=save_directory + '/' + shape_end, target_shape=shape_end, t_xsd=t_xsd, t_pc=t_pc,
                     method='DDDD')

        else:
            if shape_start + '.csv' not in os.listdir('seed_datasets'):
                print('Starting shape is incorrect')
            elif shape_end + '.csv' not in os.listdir('target_datasets'):
                print("End shape is incorrect")

        create_gifs(shape_start, shape_end)
