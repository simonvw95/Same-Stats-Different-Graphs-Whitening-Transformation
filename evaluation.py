import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.spatial import distance_matrix
from scipy.optimize import linear_sum_assignment as hung
import seaborn as sns
import os


def hungarian_scale(pos_1, pos_2):

    pos_1 = (pos_1 - np.min(pos_1)) / (np.max(pos_1) - np.min(pos_1))
    pos_2 = (pos_2 - np.min(pos_2)) / (np.max(pos_2) - np.min(pos_2))

    pdist = distance_matrix(pos_1, pos_2)
    row_idcs, col_idcs = hung(pdist)

    return pdist[row_idcs, col_idcs].sum()


def save_pdf(df, df_target, t_xsd, t_pc, save_location, target_shape = 'star_142', method = 'sgd', dp=72):

    distance = hungarian_scale(df.to_numpy(), df_target.to_numpy())
    # print(distance)

    f = open('hungarian_values.csv', 'a')
    f.write('{},{},{},{},{}\n'.format(target_shape, str(round(t_xsd, 3)), str(round(t_pc, 3)), str(round(distance, 3)), str(method)))
    f.close()

    sns.set_style('darkgrid')

    # Create figure with tight layout using GridSpec
    fig = plt.figure(figsize=(7, 6), constrained_layout=True)
    gs = fig.add_gridspec(1, 1)

    # === LEFT AXIS: SCATTERPLOT ===
    ax = fig.add_subplot(gs[0, 0])
    sns.regplot(
        x="x", y="y", data=df, ci=None, fit_reg=False,
        scatter_kws={"s": 4, "alpha": 0.9, "color": "black"},
        ax=ax
    )
    plt.ylabel('Y', fontsize=36, rotation='horizontal')
    plt.xlabel('X', fontsize=36)
    ax.tick_params(axis='both', which='major', labelsize=18)
    ax.tick_params(axis='both', which='minor', labelsize=18)
    ax.set_xlim(-25, 125)
    ax.set_ylim(-25, 125)
    ax.set_aspect("equal", adjustable="box")  # perfect square axes

    plt.savefig("{}.pdf".format(save_location), dpi=dp)
    plt.clf()
    plt.cla()
    plt.close()


# result_file = 'project/results/{}_142_xm54-265_ym47-835_xsd55-505_ysd26-935_pc-0-065/{}_142_xsd55-505_pc-0-065'.format('star', 'star')
# target_shape = 'target_datasets/{}_142'.format('star')
#
# save_pdf(result_file, target_shape, dp=72)


def temp_save(dp=72):

    datasets = ['datadance_250_{}'.format(cnt) for cnt in range(1, 6)]

    for start_dataset in datasets:
        df = pd.read_csv("target_datasets/{}.csv".format(start_dataset), header = None, names = ['x', 'y'])
        df_result = pd.read_csv('datadance_doubledozen/results/{}_{}/{}-data-00099.csv'.format(start_dataset, start_dataset, start_dataset), header = None, names = ['x', 'y'])

        sns.set_style('darkgrid')

        # Create figure with tight layout using GridSpec
        fig = plt.figure(figsize=(7, 6), constrained_layout=True)
        gs = fig.add_gridspec(1, 1)

        # === LEFT AXIS: SCATTERPLOT ===
        ax = fig.add_subplot(gs[0, 0])
        sns.regplot(
            x="x", y="y", data=df_result, ci=None, fit_reg=False,
            scatter_kws={"s": 4, "alpha": 0.9, "color": "black"},
            ax=ax
        )

        sns.set_style('darkgrid')

        plt.ylabel('Y', fontsize=36, rotation='horizontal')
        plt.xlabel('X', fontsize=36)
        ax.tick_params(axis='both', which='major', labelsize=18)
        ax.tick_params(axis='both', which='minor', labelsize=18)
        ax.set_xlim(-25, 125)
        ax.set_ylim(-25, 125)
        ax.set_aspect("equal", adjustable="box")

        plt.savefig("target_datasets/dddd_{}.pdf".format(start_dataset), dpi=dp)
        plt.clf()
        plt.cla()
        plt.close()



# import torch
# df = pd.read_csv("target_datasets/{}.csv".format('star_142'), header=None, names=['x', 'y'])
# df_result = pd.read_csv('sgd/results/random_cloud_n142_xm54-265_ym47-835_xsd16-765_ysd26-935_pc-0-065_star_142/-data-00099.csv',
#                         header=None, names=['x', 'y'])
# t_xm = 54.265
# t_ym = 47.835
# t_xsd = 16.765
# t_ysd = 26.935
# t_pc = -0.065
# distance = hungarian_scale(df.to_numpy(), project_statistics(torch.tensor(df_result.to_numpy()).float(), t_xm, t_ym, t_xsd, t_ysd, t_pc).numpy())
# # print(distance)
#
# f = open('hungarian_values.csv', 'a')
# f.write('{},{},{},{},{}\n'.format(target_shape, str(round(t_xsd, 3)), str(round(t_pc, 3)), str(round(distance, 3)), str(method)))
# f.close()
#
# sns.set_style('darkgrid')
#
# # Create figure with tight layout using GridSpec
# fig = plt.figure(figsize=(7, 6), constrained_layout=True)
# gs = fig.add_gridspec(1, 1)
#
# # === LEFT AXIS: SCATTERPLOT ===
# ax = fig.add_subplot(gs[0, 0])
# sns.regplot(
#     x="x", y="y", data=df, ci=None, fit_reg=False,
#     scatter_kws={"s": 4, "alpha": 0.9, "color": "black"},
#     ax=ax
# )
# plt.ylabel('Y', fontsize=36, rotation='horizontal')
# plt.xlabel('X', fontsize=36)
# ax.tick_params(axis='both', which='major', labelsize=18)
# ax.tick_params(axis='both', which='minor', labelsize=18)
# ax.set_xlim(-25, 125)
# ax.set_ylim(-25, 125)
# ax.set_aspect("equal", adjustable="box")  # perfect square axes
#
# plt.savefig("{}.pdf".format(save_location), dpi=dp)
# plt.clf()
# plt.cla()
# plt.close()

# df = pd.read_csv('hungarian_values.csv')
# shape = 'datadance_250_5'
# xsd = 16.765
# pc = -0.065
#
# results = {}
# for i in range(len(df)):
#     if df.iloc[i]['shape'] == shape and df.iloc[i]['xsd'] == xsd and df.iloc[i]['pc'] == pc:
#         results[df.iloc[i]['method']] = df.iloc[i]['value']
#         print(df.iloc[i]['value'], df.iloc[i]['method'])
# print(list(sorted(results, key=results.get))[0])
