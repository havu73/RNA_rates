import pandas as pd
import numpy as np
import helper
import argparse
import os

from sympy.simplify.simplify import bottom_up


def calculate_MSE(result_df, method, G):
    '''
    Calculate the mean squared error between the true h and the estimated h
    :param result_df: pd.DataFrame, the results from the solver
    :param method: str, the method used to estimate h
    :param G: int, the length of the gene
    :return: float
    '''
    true_h = result_df['true_h']
    estimated_h = result_df[method]
    # errors weighted by length of the segments/ G
    result_df['error'] = (true_h - estimated_h)**2 * (result_df['end'] - result_df['start'])/G
    return np.sum(result_df['error'])


def calculate_fold_change(result_df, method, G):
    '''
    Calculate the fold change between the true h and the estimated h, weighted by the length of the segments
    :param result_df: pd.DataFrame, the results from the solver
    :param method: str, the method used to estimate h
    :param G: int, the length of the gene
    :return: float
    '''
    true_h = result_df['true_h']
    estimated_h = result_df[method]
    # errors weighted by length of the segments/ G
    result_df['fold_change'] = np.abs((true_h - estimated_h) / true_h) * (result_df['end'] - result_df['start'])/G
    return np.sum(result_df['fold_change'])


def read_one_result_file(fn, middle_window=1):
    '''
    Read one result file and return the results as a dictionary
    :param fn: str, the filename
    :return: dict
    '''
    df = pd.read_csv(fn, sep='\t', header=0, index_col=None) # start, end, true_h and then all the diffemrent solver results
    methods = ['simpleSolver', 'bayesLinearSolver', 'bayesRBFSolver', 'logNormalSolver']
    try:
        df = df.rename(columns={'txrate': 'true_h'})
    except:
        pass
    # each file correspond to one runs of the solver, we will measure the MSE and fold change between the true h and the estimated h
    G = df.loc[df.shape[0]-1, 'end']
    middle_start = G//2 - middle_window
    middle_end = G//2 + middle_window
    region_of_interest = middle_window*2
    df = df[(df['start'] >= middle_start) & (df['end'] <= middle_end)]
    MSE_dict = pd.Series({})
    fc_dict = pd.Series({})
    for method in methods:
        MSE_dict[f'mid_wMSE_{method}'] = calculate_MSE(df, method, G=region_of_interest)
        fc_dict[f'mid_wFC_{method}'] = calculate_fold_change(df, method, G=region_of_interest)
    return MSE_dict, fc_dict

def read_all_result_files(input_folder):
    '''
    Read all the result files in the input folder and return the results as a dataframe
    :param input_folder: str, the folder where all the result files are stored
    :return: pd.DataFrame
    '''
    design_df = pd.read_csv(os.path.join(input_folder, 'design_matrix.csv'), sep='\t', header=0, index_col=None)
    for idx, row in design_df.iterrows():
        fn = os.path.join(input_folder, f'result_{idx}.txt.gz')  # row['output_fn']
        if not os.path.isfile(fn):
            continue
        MSE_dict, fc_dict = read_one_result_file(fn)
        design_df.loc[idx, MSE_dict.index] = MSE_dict
        design_df.loc[idx, fc_dict.index] = fc_dict
    print(design_df.head())
    # drop colulmns that are all nan
    design_df = design_df.dropna(axis=1, how='all')
    design_df.dropna(axis=0, how='any', inplace=True)
    return design_df

def plot1(result_df, metric='mid_wMSE', save_fn = 'wMSE.png', y_max=10):
    '''
    4 panels, each corresponds to a method. x-axis is the true_h, y-axis is the wMSE, and different h_bin should be different colors of the boxplot
    :param result_df: pd.DataFrame
    :return: None
    '''
    import seaborn as sns
    import matplotlib.pyplot as plt
    result_df['avg_h'] = result_df['G']/result_df['time_traverse_gene']
    fig, axes = plt.subplots(1, 4, figsize=(28, 6))
    methods = ['simpleSolver', 'bayesLinearSolver', 'bayesRBFSolver', 'logNormalSolver']
    for idx, method in enumerate(methods):
        sns.boxplot(data=result_df, x='avg_h', y=f'{metric}_{method}', hue='h_bin', ax=axes[idx])
        axes[idx].set_title(f'{method}')
        axes[idx].set_ylim(top=y_max, bottom=0)
    plt.tight_layout()
    plt.savefig(save_fn)
    return

def plot2(result_df, metric='mid_wMSE', save_fn = 'wMSE.png', y_max=10):
    '''
    each row: different vary_len_kb
    each columns: different methods
    x-axis is the true_h, y-axis is the wMSE, and different h_bin should be different colors of the boxplot
    :param result_df: pd.DataFrame
    :return: None
    '''
    import seaborn as sns
    import matplotlib.pyplot as plt
    vary_bin_kb = [2.e-03, 1.e-01, 2.e-01, 5.e-01, 1.e+00, 2.e+00, 3.e+00]
    fig, axes = plt.subplots(len(vary_bin_kb), 4, figsize=(28, 6*len(vary_bin_kb)))
    methods = ['simpleSolver', 'bayesLinearSolver', 'bayesRBFSolver', 'logNormalSolver']
    for row_idx, bin in enumerate(vary_bin_kb):
        for col_idx, method in enumerate(methods):
            plot_df = result_df[result_df['vary_bin_kb'] == bin]
            sns.boxplot(data=plot_df, x='true_h', y=f'{metric}_{method}', hue='h_bin', ax=axes[row_idx, col_idx])
            axes[row_idx, col_idx].set_title(f'{method}')
            axes[row_idx, col_idx].set_ylim(top=y_max, bottom=0)
            # set font size of labels
            axes[row_idx, col_idx].set_xlabel('True h', fontsize=16)
            axes[row_idx, col_idx].set_ylabel(f'{metric}', fontsize=16)
            axes[row_idx, col_idx].set_title(f'{method}, varyBin={bin}', fontsize=16)
    plt.tight_layout()
    plt.savefig(save_fn)
    return