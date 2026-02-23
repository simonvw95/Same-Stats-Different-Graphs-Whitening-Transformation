import torch
import random
import pandas as pd
import os
import pytweening
import numpy as np
from tqdm import tqdm
from datasaurus_dozen_implem import save_scatter_and_results, create_gifs
from project_implem import project_statistics
from evaluation import save_pdf


def get_statistics(coords):

    xm = torch.mean(coords[:, 0])
    ym = torch.mean(coords[:, 1])
    xsd = torch.std(coords[:, 0])
    ysd = torch.std(coords[:, 1])
    pc = torch.corrcoef(coords.T)[0][1]

    return xm, ym, xsd, ysd, pc


def error_loss(coords, t_xm = 54.265, t_ym = 47.835, t_xsd = 16.765, t_ysd = 26.935, t_pc = -0.065):

    xm, ym, xsd, ysd, pc = get_statistics(coords)

    diff_xm = (xm - t_xm) ** 2
    diff_ym = (ym - t_ym) ** 2
    diff_xsd = (xsd - t_xsd) ** 2
    diff_ysd = (ysd - t_ysd) ** 2
    diff_pc = (pc - t_pc) ** 2

    return diff_xm + diff_ym + diff_xsd + diff_ysd + diff_pc


# less precise chamfer algorithm, allows for multiple points to overlap, good for first initialization
def chamfer(pos_1, pos_2, sample_size):

    n = pos_1.shape[0]

    if sample_size:
        if sample_size <= n:
            i = torch.tensor(random.sample(range(n), sample_size), device=pos_1.device)
            pos = pos_1[i, :]
    else:
        pos = pos_1

    # scale new positions, target pos already scaled once before
    pos = scale(pos)

    pdist = torch.cdist(pos, pos_2)
    term1 = pdist.min(dim=1).values.mean()
    term2 = pdist.min(dim=0).values.mean()

    return term1 + term2


# helper function to scale all data between 0 and 1, manually scale between 20, 80
def scale(x0):

    mtx3 = (x0 - torch.min(x0)) / (torch.max(x0) - torch.min(x0))
    mtx3 *= 60
    mtx3 += 24

    return mtx3


####################################################################
# --- optimization loop ---


def optimize(init_df, target_df, name, max_iter = 10000, sample_size = 100, t_xm=54.265, t_ym=47.835, t_xsd=16.765, t_ysd=26.935, t_pc=-0.065):

    iter_bar = tqdm(range(max_iter))

    curr_coords = torch.tensor(init_df.to_numpy()).float().requires_grad_(True)
    tar_coords = scale(torch.tensor(target_df.to_numpy())).float()

    opt = torch.optim.SGD([curr_coords], lr=0.4, momentum=0.8)  # or Adam
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size = (len(curr_coords) / sample_size), gamma=0.9)

    frame_count = 0
    frames = 100
    # this is a list of frames that we will end up writing to file
    write_frames = [int(round(pytweening.linear(x) * max_iter)) for x in np.arange(0, 1, 1 / (frames - 0))]

    extras = [max_iter] * 0
    write_frames.extend(extras)

    print('\nStarting Optimization Loop')

    for epoch in iter_bar:
        opt.zero_grad()

        loss_shape = chamfer(curr_coords, tar_coords, sample_size = sample_size)
        loss_stats = 10 * error_loss(curr_coords, t_xm=t_xm, t_ym=t_ym, t_xsd=t_xsd, t_ysd=t_ysd, t_pc=t_pc)

        loss = loss_shape + loss_stats
        loss.backward()
        opt.step()
        scheduler.step()

        if epoch % 100 == 99:
            loss_full_shape = chamfer(curr_coords.detach(), tar_coords, sample_size = None)
            loss_full_stats = error_loss(curr_coords.detach(), t_xm=t_xm, t_ym=t_ym, t_xsd=t_xsd, t_ysd=t_ysd, t_pc=t_pc)
            loss_full = loss_full_shape + loss_full_stats

            iter_bar.set_postfix({'Total loss | shape loss | stats loss | ' : [loss_full.item(), loss_full_shape.item(), loss_full_stats.item()]})

        # save this chart to the file
        for x in range(write_frames.count(epoch)):
            save_df = pd.DataFrame(curr_coords.detach().numpy(), columns = ['x', 'y'])
            save_scatter_and_results(save_df, directory = name, iter = "-image-" + format(int(frame_count), '05'), dp=150, labels=["X Mean", "Y Mean", "X SD", "Y SD", "Corr."])
            save_df.to_csv(f'{name}/' + "-data-" + format(int(frame_count), '05') + ".csv", index=False, header=False)

            frame_count = frame_count + 1

    final_save_df = pd.DataFrame(curr_coords.detach().numpy(), columns = ['x', 'y'])

    save_scatter_and_results(final_save_df, directory = name, iter = 'final_preproject', dp = 72, labels = ["X Mean", "Y Mean", "X SD", "Y SD", "Corr."])

    new_coords = project_statistics(curr_coords.detach(), t_xm=t_xm, t_ym=t_ym, t_xsd=t_xsd, t_ysd=t_ysd, t_pc=t_pc, eps=1e-8)
    save_scatter_and_results(pd.DataFrame(new_coords.detach().numpy(), columns=['x', 'y']), directory = name, iter = 'final_postproject', dp=72,
                             labels=["X Mean", "Y Mean", "X SD", "Y SD", "Corr."])

    pd.DataFrame(new_coords.detach().numpy(), columns=['x', 'y']).to_csv(f'{name}/' + "-data-final.csv", index=False, header=False)

    return curr_coords


if __name__ == '__main__':

    it = 15000
    frames = 100
    sample_size = 75

    t_xm = 54.265
    t_ym = 47.835
    # t_xsds = list(range(5, 65, 5))  # add 0.505
    t_xsd = 16.765
    t_ysd = 26.935
    t_pc = -0.065
    # t_pcs = np.arange(0, 1, 0.1)
    # shape_ends = ['x_142', 'h_lines_142', 'v_lines_142', 'wide_lines_142', 'high_lines_142', 'slant_up_142',
    #               'slant_down_142', 'circle_142', 'star_142', 'down_parab_142', 'bullseye_142', 'dots_142',
    #               'datasaurus_142']
    # shape_ends = ['bike_855', 'butterfly_855', 'custom_dinosaur_855', 'maple_leaves_855', 'netherlands_855', 'palm_tree_855', 'rooster_855', 'uu_855']
    shape_ends = ['datadance_250_{}'.format(cnt) for cnt in range(1, 25)]

    # for i in t_pcs:
    for shape_end in shape_ends:
        # t_pc = round(i + 0.055, 3)  # 0.505 for xsd, 0.055 for pc
        # t_xsd = round(i + 0.505, 3)
        shape_start = 'random_cloud_n250_xm{}_ym{}_xsd{}_ysd{}_pc{}'.format(str(t_xm).replace('.', '-'),
                                                                            str(t_ym).replace('.', '-'), str(t_xsd).replace('.', '-'),
                                                                            str(t_ysd).replace('.', '-'), str(t_pc).replace('.', '-'))

        # shape_end = 'star_142'
        save_directory = f'sgd/results/{shape_start}_{shape_end}'

        if (shape_start + '.csv' in os.listdir('seed_datasets')) and (shape_end + '.csv' in os.listdir('target_datasets')):

            init_df = pd.read_csv('seed_datasets/{}.csv'.format(shape_start), index_col=0)
            target_df = pd.read_csv('target_datasets/{}.csv'.format(shape_end), header=None, names=['x', 'y'])

            # make the directory if we have to
            if os.path.exists(save_directory):
                print(f"File {save_directory} exists")
            else:
                os.mkdir(save_directory)

            opt_coords = optimize(init_df, target_df, name = save_directory, max_iter=it, sample_size=sample_size,
                                  t_xm = t_xm, t_ym = t_ym, t_xsd = t_xsd, t_ysd = t_ysd, t_pc = t_pc)

            save_pdf(df = pd.DataFrame(opt_coords.detach().numpy(), columns=['x', 'y']), df_target = target_df, save_location = save_directory + '/' + shape_start, target_shape = shape_end, t_xsd = t_xsd, t_pc = t_pc, method='sgd')

        else:
            if shape_start + '.csv' not in os.listdir('seed_datasets'):
                print('Starting shape is incorrect')
            elif shape_end + '.csv' not in os.listdir('target_datasets'):
                print("End shape is incorrect")

        create_gifs(spec_directory = 'sgd', shape_start=shape_start, shape_end = shape_end)
