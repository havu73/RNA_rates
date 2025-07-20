import os.path
import pandas as pd
import numpy as np
import glob

from pynndescent.distances import spearmanr

from debug import read_one_file, merge_dfs
'''
This file aims to answer the question: for each gene in each iteration, what the metrics of performance between pred_h and true_h?
Output: a dataframe with the following columns: gene, max_coverage, bin_h, MSE, pearsonR
'''
def get_gene_list(large_folder):
    fn_list = glob.glob(large_folder + 'iter_1/*/*gtf.gz')
    gene_list = list(map(lambda x: x.split('/')[-2], fn_list))
    return gene_list

def calculate_mse(folder, binH_idx):
    '''
    Given a folder, calculate the MSE between the pred_h and true_h
    '''
    x0_x1_fn = f'{folder}/pred_h.csv.gz'
    methods = ['simpleSmooth', 'simpleSolver', 'bayesLinearSolver', 'bayesRBFSolver', 'logNormalSolver']
    results = {}
    try:
        df = pd.read_csv(x0_x1_fn, sep='\t')
        df = df[df['index'] == binH_idx]
        for method in methods:
            pred_h = df[method]
            nonnan_idx = ~np.isnan(pred_h)
            pred_h = pred_h[nonnan_idx]
            true_h = df['true_h'][nonnan_idx]
            mse = ((pred_h - true_h)**2).mean()
            results[method] = mse
    except:
        results = {method: np.nan for method in methods}
    return results

def calculate_pearsonr(folder, binH_idx):
    '''

    :param folder:
    :param binH_idx:
    :return:
    '''
    x0_x1_fn = f'{folder}/pred_h.csv.gz'
    methods = ['simpleSmooth', 'simpleSolver', 'bayesLinearSolver', 'bayesRBFSolver', 'logNormalSolver']
    results = {}
    try:
        df = pd.read_csv(x0_x1_fn, sep='\t')
        df = df[df['index'] == binH_idx]
        for method in methods:
            pred_h = df[method]
            nonnan_idx = ~np.isnan(pred_h)
            pred_h = pred_h[nonnan_idx]
            true_h = df['true_h'][nonnan_idx]
            pearsonR = np.corrcoef(pred_h, true_h)[0,1]
            results[method] = pearsonR
    except:
        results = {method: np.nan for method in methods}
    return results

def calculate_spearmanR(folder, binH_idx):
    '''

       :param folder:
       :param binH_idx:
       :return:
       '''
    x0_x1_fn = f'{folder}/pred_h.csv.gz'
    methods = ['simpleSmooth', 'simpleSolver', 'bayesLinearSolver', 'bayesRBFSolver', 'logNormalSolver']
    results = {}
    try:
        df = pd.read_csv(x0_x1_fn, sep='\t')
        df = df[df['index'] == binH_idx]
        for method in methods:
            pred_h = df[method]
            nonnan_idx = ~np.isnan(pred_h)
            pred_h = pred_h[nonnan_idx]
            true_h = df['true_h'][nonnan_idx]
            spearmanR = np.corrcoef(pred_h, true_h, method='spearman')[0,1]
            results[method] = spearmanR
    except:
        results = {method: np.nan for method in methods}
    return results

def calculate_avg_trueH(folder, gene_name):
    '''
    Given a folder, calculate the average true_h of the gene
    '''
    x0_x1_fn = f'{folder}/pred_h.csv.gz'
    methods = ['simpleSmooth', 'simpleSolver', 'bayesLinearSolver', 'bayesRBFSolver', 'logNormalSolver']
    results = {}
    try:
        df = pd.read_csv(x0_x1_fn, sep='\t')
        df = df[df['index'] == binH_idx]
        # calculate the average trueH and average predH for the nonnan valules
        for method in methods:
            pred_h = df[method]
            nonnan_idx = ~np.isnan(pred_h)
            pred_h = pred_h[nonnan_idx]
            true_h = df['true_h'][nonnan_idx]
            avg_trueH = true_h.mean()
            results[method] = avg_trueH
    except:
        results = {method: np.nan for method in methods}
    return results

if __name__ == '__main__':
    large_folder = '/gladstone/engelhardt/lab/hvu/RNA_rates/data_from_jesse/ha_reorganized_data/'
    time =5
    '''
    large_folder
    |__ iter_1
    |   |__ gene
    |       |__ gene.gtf.gz
    '''
    gene_list = get_gene_list(large_folder)
    num_iterations = 100
    result_df = pd.DataFrame(columns = ['gene', 'iteration', 'bin_h', 'method', 'MSE', 'pearsonR', 'spearmanR'])
    binH_dict={0: 0.1, 1:0.5, 2:1, 3:2}  # index, bin of elongation rate in kb
    for i in [1,2]: #range(1,num_iterations+1):
        for gene in gene_list:
            folder = f'{large_folder}/iter_{i}/{gene}'
            for binH_idx, binH in binH_dict.items():
                mse_results = calculate_mse(folder, binH_idx)
                pearsonR_results = calculate_pearsonr(folder, binH_idx)
                spearmanr_results = calculate_spearmanR(folder, binH_idx)
                for method, mse in mse_results.items():
                    result_df.loc[result_df.shape[0]] = pd.Series({'gene': gene, 'iteration': i, 'bin_h': binH, 'method': method, 'MSE': mse, 'pearsonR': pearsonR_results[method], 'spearmanR': spearmanr_results[method]})
    result_df.to_csv(f'{large_folder}/metrics_trueH_predH.csv.gz', index=False, sep = '\t', compression='gzip')
    print('Done')
