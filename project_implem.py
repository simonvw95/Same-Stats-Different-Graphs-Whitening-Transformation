import torch
import pandas as pd
import os
import numpy as np
from evaluation import save_pdf
from datasaurus_dozen_implem import save_scatter_and_results


def project_statistics(coords, t_xm=54.265, t_ym=47.835, t_xsd=16.765, t_ysd=26.935, t_pc=-0.065, eps=1e-8):
    device = coords.device

    # center the current scatterplot
    mean = coords.mean(dim=0, keepdim=True)
    X = coords - mean

    # covariance matrix of the current scatterplot
    cov = torch.cov(X.T)

    # target covariance matrix built from the target summary statistics
    target_cov = torch.tensor([
        [t_xsd**2, t_pc * t_xsd * t_ysd],
        [t_pc * t_xsd * t_ysd, t_ysd**2]
    ], device=device).float()

    # whitening transform
    eigvals, eigvecs = torch.linalg.eigh(cov + eps * torch.eye(2, device=device))
    W = eigvecs @ torch.diag(1.0 / torch.sqrt(eigvals)) @ eigvecs.T

    # coloring transform
    teigvals, teigvecs = torch.linalg.eigh(target_cov)
    C = teigvecs @ torch.diag(torch.sqrt(teigvals)) @ teigvecs.T

    # apply the whitening and coloring transform
    X_proj = (X @ W.T) @ C.T
    # apply the target x and y mean
    X_proj += torch.tensor([t_xm, t_ym], device=device)

    return X_proj


def create_random_cloud(t_xm=54.265, t_ym=47.835, t_xsd=16.765, t_ysd=26.935, t_pc=-0.065):

    init_df = pd.read_csv('seed_datasets/random_cloud_142.csv', index_col=0)
    new_coords = project_statistics(torch.tensor(init_df.to_numpy()).float(), t_xm, t_ym, t_xsd, t_ysd, t_pc)
    new_df = pd.DataFrame(new_coords, columns=['x', 'y'])
    new_df.to_csv('seed_datasets/random_cloud_n142_xm{}_ym{}_xsd{}_ysd{}_pc{}.csv'
                  .format(str(t_xm).replace('.', '-'), str(t_ym).replace('.', '-'), str(t_xsd).replace('.', '-'),
                          str(t_ysd).replace('.', '-'),  str(t_pc).replace('.', '-')))


if __name__ == '__main__':

    t_xm = 54.265
    t_ym = 47.835
    # t_xsds = list(range(5, 65, 5))  # add 0.505
    t_xsd = 16.765
    t_ysd = 26.935
    t_pc = -0.065
    # t_pcs = np.arange(0, 1, 0.1)

    # shape_ends = ['bike_855', 'butterfly_855', 'custom_dinosaur_855', 'maple_leaves_855', 'netherlands_855', 'palm_tree_855', 'rooster_855', 'uu_855']
    # shape_ends = ['x_142', 'h_lines_142', 'v_lines_142', 'wide_lines_142', 'high_lines_142', 'slant_up_142',                    'slant_down_142', 'circle_142', 'star_142', 'down_parab_142', 'bullseye_142', 'dots_142', 'datasaurus_142']
    shape_ends = ['datadance_250_{}'.format(cnt) for cnt in range(1, 25)]

    for shape_end in shape_ends:
    # for i in t_xsds:
        # t_pc = round(i + 0.055, 3)  # 0.505 for xsd, 0.055 for pc
        # t_xsd = round(i + 0.505, 3)

        # shape_end = 'star_142'
        save_directory = f'project/results/{shape_end}' + '_xm{}_ym{}_xsd{}_ysd{}_pc{}'\
            .format(str(t_xm), str(t_ym), str(t_xsd), str(t_ysd), str(t_pc)).replace('.', '-')

        if shape_end + '.csv' in os.listdir('target_datasets'):

            target_df = pd.read_csv('target_datasets/{}.csv'.format(shape_end), header=None, names=['x', 'y'])

            # make the directory if we have to
            if os.path.exists(save_directory):
                print(f"File {save_directory} exists")
            else:
                os.mkdir(save_directory)

            curr_coords = torch.tensor(target_df.to_numpy()).float()

            new_coords = project_statistics(curr_coords.detach(), t_xm=t_xm, t_ym=t_ym, t_xsd=t_xsd, t_ysd=t_ysd,
                                            t_pc=t_pc, eps=1e-8)

            save_scatter_and_results(pd.DataFrame(new_coords.detach().numpy(), columns=['x', 'y']), directory=save_directory,
                                     iter=shape_end + '_xsd{}_pc{}'.format(str(t_xsd).replace('.', '-'), str(t_pc).replace('.', '-')), dp=72,
                                     labels=["X Mean", "Y Mean", "X SD", "Y SD", "Corr."])

            pd.DataFrame(new_coords.detach().numpy(), columns=['x', 'y']).to_csv("{}/{}.csv".format(save_directory, shape_end + '_xsd{}_pc{}'.format(str(t_xsd).replace('.', '-'), str(t_pc).replace('.', '-'))))

            save_pdf(df = pd.DataFrame(new_coords.detach().numpy(), columns=['x', 'y']), df_target = target_df, save_location = save_directory + '/' + shape_end + '_xsd{}_pc{}'.format(str(t_xsd).replace('.', '-'), str(t_pc).replace('.', '-')), target_shape = shape_end, t_xsd = t_xsd, t_pc = t_pc, method='transformation')

        else:
            if shape_end + '.csv' not in os.listdir('target_datasets'):
                print("End shape is incorrect")
