import pandas as pd
import numpy as np
import helper
import argparse
import os

def question1_plot1(result_df, metric='wMSE', save_fn = None, y_max=36):
    '''
    This is a heatmap with:
    - x-axis: different values of true_h
    - y-axis: different values of h_bin
    - heatmap: values of wMSE/wFC
    :param result_df:
    :param metric:
    :param save_fn:
    :param y_max:
    :return:
    '''
    import seaborn as sns
    import matplotlib.pyplot as plt
    result_df['true_h'] = result_df['G']/result_df['time_traverse_gene']
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    sns.heatmap(data=result_df.pivot(index='h_bin', columns='true_h', values=f'{metric}_simpleSolver'), ax=ax)
    ax.set_title(f'{metric} for simpleSolver')
    plt.tight_layout()
    if save_fn:
        plt.savefig(save_fn)
    return

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


def read_one_result_file(fn, methods = ['simpleSmooth', 'simpleSolver', 'bayesLinearSolver', 'bayesRBFSolver', 'logNormalSolver']):
    '''
    Read one result file and return the results as a dictionary
    :param fn: str, the filename
    :return: dict
    '''
    df = pd.read_csv(fn, sep='\t', header=0, index_col=None) # start, end, true_h and then all the diffemrent solver results
    try:
        df = df.rename(columns={'txrate': 'true_h'})
    except:
        pass
    # each file correspond to one runs of the solver, we will measure the MSE and fold change between the true h and the estimated h
    G = df.loc[df.shape[0]-1, 'end']
    MSE_dict = pd.Series({})
    fc_dict = pd.Series({})
    for method in methods:
        MSE_dict[f'wMSE_{method}'] = calculate_MSE(df, method, G)
        fc_dict[f'wFC_{method}'] = calculate_fold_change(df, method, G)
    return MSE_dict, fc_dict

def read_one_result_raw(fn, summ_df, methods = ['simpleSolver', 'simpleSmooth', 'bayesLinearSolver', 'bayesRBFSolver', 'logNormalSolver']):
    '''
    Read one result file and return the results as a dictionary
    :param fn: str, the filename
    :return: dict
    '''
    df = pd.read_csv(fn, sep='\t', header=0,
                     index_col=None)  # start, end, true_h and then all the diffemrent solver results
    file_index = int(fn.split('/')[-1].split('.')[0].split('_')[-1])
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



def read_all_result_files(input_folder, methods = ['simpleSmooth', 'simpleSolver', 'bayesLinearSolver', 'bayesRBFSolver', 'logNormalSolver']):
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
        MSE_dict, fc_dict = read_one_result_file(fn, methods=methods)
        design_df.loc[idx, MSE_dict.index] = MSE_dict
        design_df.loc[idx, fc_dict.index] = fc_dict
    print(design_df.head())
    # drop colulmns that are all nan
    design_df = design_df.dropna(axis=1, how='all')
    design_df.dropna(axis=0, how='any', inplace=True)
    return design_df

def draw_results_compare_methods(result_df):
    '''
    Regardless all the other conditions, just combine all the result for different methods, and plot the boxplot of the wMSE and wFC
    :param result_df: pd.DataFrame
    :return: None
    '''
    import seaborn as sns
    import matplotlib.pyplot as plt
    # draw boxplots of the wMSE and wFC for each method --> 2 plots
    metric = 'wMSE'
    fig, ax = plt.subplots(1, 2, figsize=(20, 10))
    methods = [col for col in result_df.columns if metric in col]
    melt_df = result_df[methods].melt(var_name='Method', value_name=metric)
    sns.boxplot(data=melt_df, ax=ax[0], x='Method', y=metric)
    metric = 'wFC'
    methods = [col for col in result_df.columns if metric in col]
    melt_df = result_df[methods].melt(var_name='Method', value_name=metric)
    sns.boxplot(data=melt_df, ax=ax[1], x='Method', y=metric)
    plt.savefig('results_comparison.png')
    plt.close()
    return


def plot1(result_df, metric='wMSE', save_fn = 'wMSE.png', y_max=36):
    '''
    4 panels, each corresponds to a method. x-axis is the true_h, y-axis is the wMSE, and different h_bin should be different colors of the boxplot
    :param result_df: pd.DataFrame
    :return: None
    '''
    import seaborn as sns
    import matplotlib.pyplot as plt
    result_df['true_h'] = result_df['G']/result_df['time_traverse_gene']
    fig, axes = plt.subplots(1, 4, figsize=(28, 6))
    methods = ['simpleSolver', 'bayesLinearSolver', 'bayesRBFSolver', 'logNormalSolver']
    for idx, method in enumerate(methods):
        sns.boxplot(data=result_df, x='true_h', y=f'{metric}_{method}', hue='h_bin', ax=axes[idx])
        axes[idx].set_title(f'{method}')
        axes[idx].set_ylim(top=y_max)
    plt.tight_layout()
    plt.savefig(save_fn)
    return

def plot2(result_df, metric='wMSE', save_fn = 'wMSE.png', y_max=10):
    '''
    each panel corresponds to a different value of true_h,
    x_axis: different methods, based on the order of the methods
    y_axis: wMSE/wFC
    hue: h_bin
    :param result_df: pd.DataFrame
    :return: None
    '''
    import seaborn as sns
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(4, 4, figsize=(36, 36))
    result_df['true_h'] = result_df['G']/result_df['time_traverse_gene']
    methods = ['simpleSolver', 'bayesLinearSolver', 'bayesRBFSolver', 'logNormalSolver']
    true_h_list = np.unique(result_df['true_h'])  # it is sorted in increasing order by default
    for idx, true_h in enumerate(true_h_list):
        ax = axes[idx//4, idx%4]
        plot_df = result_df[result_df['true_h'] == true_h]
        plot_df = plot_df.melt(id_vars=['h_bin'], value_vars=[f'{metric}_{method}' for method in methods], var_name='method', value_name=metric)
        plot_df['method'] = plot_df['method'].apply(lambda x: x.split('_')[1])  # from wFC_method to method
        max_metric = plot_df[metric].max()
        if max_metric > y_max:
            ax.set_ylim(top=y_max)
        sns.boxplot(data=plot_df, x='method', y=f'{metric}', hue='h_bin', ax=ax)
        ax.set_title(f'true_h={true_h}, {metric}')
    plt.tight_layout()
    plt.savefig(save_fn)
    return


def plot4(result_df, metric='wMSE', output_folder = './', y_max=36):
    '''
    each panel corresponds to a different value of true_h,
    x_axis: clustering of different labeling-time and different methods
    y_axis: wMSE/wFC
    hue: h_bin
    :param result_df: pd.DataFrame
    :return: None
    '''
    import seaborn as sns
    import matplotlib.pyplot as plt
    methods = ['simpleSolver', 'bayesLinearSolver', 'bayesRBFSolver', 'logNormalSolver']
    if 'true_h' not in result_df.columns:
        result_df['true_h'] = result_df['G']/result_df['time_traverse_gene']
    true_h_list = np.unique(result_df['true_h'])  # it is sorted in increasing order by default
    for idx, true_h in enumerate(true_h_list):
        plot_df = result_df[result_df['true_h'] == true_h]
        plot_df = plot_df.melt(id_vars=['h_bin', 'label_time'], value_vars=[f'{metric}_{method}' for method in methods], var_name='method', value_name=metric)
        plot_df['method'] = plot_df['method'].apply(lambda x: x.split('_')[1])  # from wFC_method to method
        max_metric = plot_df[metric].max()
        if max_metric < y_max:
            y_max = max_metric
        fig_width = 6  # Width of the figure
        fig_height = 3  # Height of the figure
        aspect = fig_width / fig_height
        grid = sns.catplot(
            x='method',
            y=metric,
            hue='h_bin',
            col='label_time',
            data=plot_df,
            kind='box',
            height=fig_height,
            aspect=aspect
        )
        grid.set(ylim=(0, y_max))
        grid.fig.suptitle(f'true_h={true_h}, {metric}')
        plt.tight_layout()
        save_fn = os.path.join(output_folder, f'{metric}_true_h_{true_h}.png')
        plt.savefig(save_fn)
    return

def plot3(result_df, metric='wMSE', save_fn = 'wMSE.png', y_max=36):
    '''
    4 panels, each corresponds to a method. x-axis is the true_h, y-axis is the wMSE, and different h_bin should be different colors of the boxplot
    :param result_df: pd.DataFrame
    :return: None
    '''
    import seaborn as sns
    import matplotlib.pyplot as plt
    result_df['true_h'] = result_df['G']/result_df['time_traverse_gene']
    fig, axes = plt.subplots(1, 4, figsize=(28, 6))
    methods = ['simpleSolver', 'bayesLinearSolver', 'bayesRBFSolver', 'logNormalSolver']
    for idx, method in enumerate(methods):
        sns.boxplot(data=result_df, x='true_h', y=f'{metric}_{method}', hue='h_bin', ax=axes[idx])
        axes[idx].set_title(f'{method}')
        axes[idx].set_ylim(top=y_max)
    plt.tight_layout()
    plt.savefig(save_fn)
    return

def plot5(result_df, save_fn=None):
    '''
    x-axis: length along the gene
    y-axis: different methods's prediction
    :param result_fn: str, the filename of the result file
    :return: None
    '''
    import seaborn as sns
    import matplotlib.pyplot as plt
    result_df['middle'] = (result_df['start'] + result_df['end'])/2
    result_df.rename(columns={'txrate': 'true_h'}, inplace=True)
    methods = ['simpleSolver', 'bayesLinearSolver', 'bayesRBFSolver', 'logNormalSolver', 'true_h']
    fig, axes = plt.subplots(2, 6, figsize=(36,12))
    for row_idx, vary_bin in enumerate([0.002, 0.01]):
        for col_idx, h_bin in enumerate([1.e-03, 1.e-02, 1.e-01, 1.e+00, 2.e+00, 5.e+00]):
            ax = axes[row_idx, col_idx]
            plot_df = result_df[(result_df['h_bin'] == h_bin) & (result_df['vary_bin'] == vary_bin)]
            sns.lineplot(data=plot_df, x='middle', y='true_h', label='true_h', ax=ax)
            sns.lineplot(data=plot_df, x='middle', y='simpleSolver', label='simpleSolver', ax=ax)
            sns.lineplot(data=plot_df, x='middle', y='bayesLinearSolver', label='bayesLinearSolver', ax=ax)
            sns.lineplot(data=plot_df, x='middle', y='bayesRBFSolver', label='bayesRBFSolver', ax=ax)
            sns.lineplot(data=plot_df, x='middle', y='logNormalSolver', label='logNormalSolver', ax=ax)
            ax.set_title(f'vary_bin={vary_bin}, h_bin={h_bin}', fontsize=16)
            #set the font size of the axis to be bigger
            ax.tick_params(axis='both', which='major', labelsize=16)

    plt.tight_layout()

    if save_fn:
        plt.savefig(save_fn)
    return

def plot6(result_df, save_fn):
    '''
    x-axis: length along the gene
    :param result_df:
    :param save_fn:
    :return:
    '''
if __name__ == '__main__':
    parser = argparse.ArgumentParser('This script is meant to be summarize the results from the solver analysis')
    parser.add_argument('--input_folder', required=True, type=str, help='Where all the result_{job_id}.txt.gz files are stored')
    parser.add_argument('--output_folder', required=True, type=str, help='Where to store the output file')
    args = parser.parse_args()
    helper.make_dir(args.output_folder)
    print('Done getting command line arguments')
    design_df = pd.read_csv('vary_middle_summary.txt.gz', header = 0, index_col =None, sep = '\t') #read_all_result_files(args.input_folder)
    methods = ['simpleSolver', 'bayesLinearSolver', 'bayesRBFSolver', 'logNormalSolver']
    # draw the results such that we have 4 axes for each method.
    # for each method, x-axis should be the true_h, and y-axis should be the wMSE, and different h_bin should be different colors of the boxplot
    plot1(design_df, metric='wMSE', save_fn = os.path.join(args.output_folder, 'wMSE.png'), y_max=36)
    plot1(design_df, metric='wFC', save_fn = os.path.join(args.output_folder, 'wFC.png'), y_max=2)
    plot4(design_df, metric='wMSE', output_folder=os.path.join(args.output_folder))
    plot4(design_df, metric='wFC', output_folder=os.path.join(args.output_folder))
