import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def calculate_MSE(result_df, method, G):
    '''
    Calculate the mean squared error between the true h and the estimated h
    :param result_df: pd.DataFrame, the results from the solver
    :param method: str, the method used to estimate h
    :param G: int, the length of the gene
    :return: float
    '''
    true_h = result_df['true_h']
    result_df.loc[result_df.shape[0]-1, 'end'] = G
    estimated_h = result_df[method]
    # errors weighted by length of the segments/ G
    result_df['error'] = (true_h - estimated_h)**2 * (result_df['end'] - result_df['start'])/G
    return np.sqrt(np.sum(result_df['error']))


def calculate_fold_change(result_df, method, G):
    '''
    Calculate the fold change between the true h and the estimated h, weighted by the length of the segments
    :param result_df: pd.DataFrame, the results from the solver
    :param method: str, the method used to estimate h
    :param G: int, the length of the gene
    :return: float
    '''
    true_h = result_df['true_h']
    result_df.loc[result_df.shape[0] - 1, 'end'] = G
    estimated_h = result_df[method]
    # errors weighted by length of the segments/ G
    result_df['fold_change'] = np.abs((true_h - estimated_h) / true_h) * (result_df['end'] - result_df['start'])/G
    return np.sum(result_df['fold_change'])


def read_one_result_file(fn, G = None):
    '''
    Read one result file and return the results as a dictionary
    :param fn: str, the
    :param G: int, the length of the gene
    :return: dict
    '''
    try:
        df = pd.read_csv(fn, sep='\t', header=0, index_col=None) # start, end, true_h and then all the diffemrent solver results
    except:
        return None, None, None
    methods = ['simpleSolver', 'bayesLinearSolver', 'bayesRBFSolver', 'logNormalSolver']
    try:
        df = df.rename(columns={'txrate': 'true_h'})
    except:
        pass
    # each file correspond to one runs of the solver, we will measure the MSE and fold change between the true h and the estimated h
    if G is None:
        G = df.loc[df.shape[0]-1, 'end']
    MSE_dict = pd.Series({})
    fc_dict = pd.Series({})
    values_dict = pd.Series({})
    for method in methods:
        MSE_dict[f'wMSE_{method}'] = calculate_MSE(df, method, G)
        fc_dict[f'wFC_{method}'] = calculate_fold_change(df, method, G)
        values_dict[f'{method}'] = df[method].values[0]
    values_dict['true_h'] = df['true_h'].values[0]
    return MSE_dict, fc_dict, values_dict

def read_one_result_raw(fn, summ_df):
    '''
    Read one result file and return the results as a dictionary
    :param fn: str, the filename
    :return: dict
    '''
    df = pd.read_csv(fn, sep='\t', header=0,
                     index_col=None)  # start, end, true_h and then all the diffemrent solver results
    file_index = int(fn.split('/')[-1].split('.')[0].split('_')[-1])
    methods = ['simpleSolver', 'bayesLinearSolver', 'bayesRBFSolver', 'logNormalSolver']
    try:
        df = df.rename(columns={'txrate': 'true_h'})
    except:
        pass
    h_bin =summ_df.loc[file_index, 'h_bin']
    vary_bin_kb = summ_df.loc[file_index, 'vary_bin_kb']
    vary_fold = summ_df.loc[file_index, 'vary_fold']
    df['h_bin'] = h_bin
    df['vary_bin'] = vary_bin_kb
    df['vary_fold'] = vary_fold
    return df



def read_all_result_files(input_folder):
    '''
    Read all the result files in the input folder and return the results as a dataframe
    :param input_folder: str, the folder where all the result files are stored
    :return: pd.DataFrame
    '''
    design_df = pd.read_csv(os.path.join(input_folder, 'design_matrix.csv'), sep='\t', header=0, index_col=None)
    for idx, row in design_df.iterrows():
        fn = os.path.join(input_folder, f'result_{idx}', 'pred_h.csv.gz')  # row['output_fn']
        if not os.path.isfile(fn):
            continue
        MSE_dict, fc_dict, values_dict = read_one_result_file(fn, G= row['G'])
        if MSE_dict is None:
            continue
        design_df.loc[idx, MSE_dict.index] = MSE_dict
        design_df.loc[idx, fc_dict.index] = fc_dict
        design_df.loc[idx, values_dict.index] = values_dict
    print(design_df.head())
    # drop colulmns that are all nan
    design_df = design_df.dropna(axis=1, how='all')
    design_df.dropna(axis=0, how='any', inplace=True)
    return design_df

def plot1(design_df, save_fn = ''):
    '''
    x-axis: true_h
    y_axis: different methods' prediction
    colors: different methods
    :param design_df: pd.DataFrame, the design matrix
    :param save_fn: str, the filename to save the plot
    :return: None
    '''
    # draw the plot where x-axis: true elongation rate, y-axis: estimated elongation rate
    # color of dots: different methods
    methods = ['simpleSolver', 'bayesLinearSolver', 'logNormalSolver']
    plot_df = design_df[methods + ['true_h']].copy()
    plot_df = plot_df.dropna()
    # pivot the dataframe to have columns: true_h, method, estimated_h
    plot_df = plot_df.melt(id_vars=['true_h'], value_vars=methods, var_name='method', value_name='estimated_h')
    fig, ax = plt.subplots()
    sns.scatterplot(data=plot_df, x='true_h', y='estimated_h', hue='method', ax=ax, alpha=0.3)
    # draw the identity line
    x = np.linspace(0, 13, 100)
    ax.plot(x, x, color='black')
    return

def plot2(design_df, metric = 'wMSE', save_fn=None):
    '''
    Question: Does gene expression affect predictions?
    x-axis: gene expression
    y-axis: wMSE of (pred_h, true_h)
    boxplots
    :param design_df:
    :param save_fn:
    :return:
    '''
    methods = ['simpleSolver', 'bayesLinearSolver', 'logNormalSolver']
    design_df['trans_per_min'] = design_df['lambda_init'] * design_df['burst_size']
    columns_to_plot = [f'{metric}_{method}' for method in methods] + ['trans_per_min']
    # create a melted df with trans_per_min, wMSE, method
    plot_df = design_df[columns_to_plot].copy()
    plot_df = plot_df.melt(id_vars=['trans_per_min'], value_vars=[f'{metric}_{method}' for method in methods], var_name='method', value_name=f'{metric}')
    print(plot_df.head())
    fig, ax = plt.subplots()
    sns.boxplot(data=plot_df, x='tras_per_min', y=f'{metric}', hue='method', ax=ax)
    return


