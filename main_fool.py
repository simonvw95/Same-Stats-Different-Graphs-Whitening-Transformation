import argparse
import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import numpy as np
from datasaurus_dozen_implem import get_values
from project_implem import project_statistics
# from datasaurus_dozen_implem import save_scatter_and_results


class CustomRange(object):
    # from https://stackoverflow.com/a/59678681
    def __init__(self, start, end):
        self.start = start
        self.end = end

    def __eq__(self, other):
        return self.start <= other <= self.end

    def __contains__(self, item):
        return self.__eq__(item)

    def __iter__(self):
        yield self

    def __repr__(self):
        return '[{0},{1}]'.format(self.start, self.end)


parser = argparse.ArgumentParser()

# parser.add_argument("start", help="Starting (seed) dataset (default=random cloud, not relevant for the linear transformation)", type=str, default='random_cloud')
parser.add_argument("-t", "--target", help="Target dataset (default=star_142)", type=str, const='star_142', nargs='?')
# parser.add_argument("method", help="Method used for fooling (default=linear transformation)", type=str, default='linear transformation')

parser.add_argument("-xm", "--XMean", help="Float of the specified x mean [-100,100] (default=54.265)",
                    type=float, const=54.265, nargs='?', choices=CustomRange(-100, 100))
parser.add_argument("-ym", "--YMean", help="Float of the specified Y mean [-100,100] (default=47.835)",
                    type=float, const=47.835, nargs='?', choices=CustomRange(-100, 100))
parser.add_argument("-xsd", "--XStDev", help="Float of the specified X standard deviation [-50,50] (default=16.765)",
                    type=float, const=16.765, nargs='?', choices=CustomRange(-50, 50))
parser.add_argument("-ysd", "--YStDev", help="Float of the specified Y standard deviation [-50,50] (default=26.935)",
                    type=float, const=26.935, nargs='?', choices=CustomRange(-50, 50))
parser.add_argument("-pc", "--PearCorr", help="Float of the specified Pearson Correlation [-1,1] (default=-0.065)",
                    type=float, const=-0.065, nargs='?', choices=CustomRange(-1, 1))

# parser.add_argument("acc", help="Integer of the number of decimals we should keep the same [0,3] (default=2)", type=float, default=2)
args = parser.parse_args()
t_xm = args.XMean
t_ym = args.YMean
t_xsd = args.XStDev
t_ysd = args.YStDev
t_pc = args.PearCorr

save_directory = r'project/results/commandline_{}_xm{}_ym{}_xsd{}_ysd{}_pc{}'\
                 .format(args.target, str(t_xm), str(t_ym), str(t_xsd), str(t_ysd), str(t_pc)).replace('.', '-')

if args.target + '.csv' in os.listdir('target_datasets'):
    print('Target dataset accepted')

    # load target dataframe
    target_df = pd.read_csv('target_datasets/{}.csv'.format(args.target), header=None, names=['x', 'y'])
    # make the directory if we have to
    if os.path.exists(save_directory):
        print(f"File {save_directory} exists")
    else:
        os.mkdir(save_directory)

    # load the current coordinates
    curr_coords = torch.tensor(target_df.to_numpy()).float()

    # transform them to the target shape
    start_time = time.time()
    new_coords = project_statistics(curr_coords.detach(), t_xm=t_xm, t_ym=t_ym, t_xsd=t_xsd, t_ysd=t_ysd, t_pc=t_pc, eps=1e-8)
    end_time = time.time()

    print('Done transforming the dataset to the target shape. Done in: {} seconds'.format(str(round(end_time - start_time, 6))))

    # put them in a new dataframe
    df = pd.DataFrame(new_coords.detach().numpy(), columns=['x', 'y'])
    df.to_csv("{}/{}.csv".format(save_directory, args.target))

    # save the scatter plot as a png
    res = get_values(df)
    fs = 30
    labels = ["X Mean", "Y Mean", "X SD", "Y SD", "Corr."]
    max_label_length = max(len(la) for la in labels)

    # create figure with tight layout using GridSpec
    fig = plt.figure(figsize=(12, 5), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 0.8])

    # === LEFT AXIS: SCATTERPLOT ===
    ax = fig.add_subplot(gs[0, 0])
    sns.regplot(
        x="x", y="y", data=df, ci=None, fit_reg=False,
        scatter_kws={"s": 4, "alpha": 0.9, "color": "black"},
        ax=ax
    )

    # find the mins and maxs plus a margin of 15
    min_x, min_y = np.min(df.to_numpy(), axis=0) - 15
    max_x, max_y = np.max(df.to_numpy(), axis=0) + 15

    default_min = -25
    default_max = 125

    # increase the margins by 5 if the dataset contains values bigger than the default margins (preserve squareness of the xlim and ylim)
    while True:
        if min_x <= default_min or min_y <= default_min or max_x >= default_max or max_y >= default_max:
            default_min -= 5
            default_max += 5
        else:
            break

    ax.set_xlim(default_min, default_max)
    ax.set_ylim(default_min, default_max)
    ax.set_aspect("equal", adjustable="box")  # perfect square axes

    # === RIGHT AXIS: TEXT BLOCK ===
    ax_text = fig.add_subplot(gs[0, 1])
    ax_text.axis("off")

    y_positions = [0.9, 0.75, 0.60, 0.45, 0.30]

    # shadow / lighter text
    for i, (label, value) in enumerate(zip(labels, res)):
        ax_text.text(
            0.0, y_positions[i],
            label.ljust(max_label_length) + ": " + format(value, "0.9f")[:-2],
            fontsize=fs, alpha=0.3, transform=ax_text.transAxes
        )

    # main bold text
    for i, (label, value) in enumerate(zip(labels, res)):
        ax_text.text(
            0.0, y_positions[i],
            label.ljust(max_label_length) + ": " + format(value, "0.9f")[:-7],
            fontsize=fs, alpha=1, transform=ax_text.transAxes
        )

    plt.savefig("{}/{}".format(save_directory, args.target + '.png'), dpi=72)
    plt.clf()
    plt.cla()
    plt.close()

    print('Saved the new dataset as a .csv and .png')
else:
    print('Target dataset not recognized, make sure the correct .csv file is in /target_datasets/ (format {name}_{n_points}.csv)')
