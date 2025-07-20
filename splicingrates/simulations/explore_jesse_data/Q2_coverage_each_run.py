import os.path
import pandas as pd
import numpy as np
import glob
from debug import read_one_file, merge_dfs
'''
This file aims to answer the question: for each gene in each iteration, what is the coverage of the gene?

'''
def get_gene_list(large_folder):
    fn_list = glob.glob(large_folder + 'iter_1/*/*gtf.gz')
    gene_list = list(map(lambda x: x.split('/')[-2], fn_list))
    return gene_list

def calculate_max_coverage(folder, gene_name):
    '''
    Given a folder, calculate the max coverage of the gene
    '''
    fn1 = f'{folder}/{gene_name}_nt_coverage_0-5.bed.gz'
    fn2 = f'{folder}/{gene_name}_nt_coverage_5-10.bed.gz'
    fn3 = f'{folder}/{gene_name}_nt_coverage_10-15.bed.gz'
    try:
        df1, strand = read_one_file(fn1, 0)
        df2, strand = read_one_file(fn2, 1)
        df3, strand = read_one_file(fn3, 2)
        df = merge_dfs([df1, df2, df3])
        df['total_coverage'] = df[0] + df[1] + df[2]
        max_coverage = df['total_coverage'].max()
        return max_coverage
    except:
        return np.nan

def calculate_true_avg_elongRate(folder, gene_name):
    '''
    Given a folder, calculate the true average elongation rate of the gene. True average elongation rate is the length of gene / time to transcribe the gene
    :param folder:
    :param gene_name:
    :return:
    '''
    gtf_fn = f'{folder}/{gene_name}.gtf.gz'
    print(gtf_fn)
    df = pd.read_csv(gtf_fn, sep='\t', header=0) # columns chromosome      source  region_number  nucleotide_coord  time_for_this_nt  rate_for_this_nt  rate_change_per_nt
    gene_length = df['nucleotide_coord'].max()
    time_to_transcribe = df['time_for_this_nt'].sum()
    true_avg_elongRate = gene_length / time_to_transcribe
    h_kb = true_avg_elongRate / 1000
    return h_kb

def calculate_pred_avg_elongRate(folder, gene_name, time=5):
    '''
    Given a folder, calculate the predicted average elongation rate of the gene. Predicted average elongation rate is the length
    this is the PREDICTED average elongation rate, menaing that we are extracting the gene-wide avg elongation rate based on the predicted x0_x1.csv.gz file
    '''
    x0_x1_fn = f'{folder}/x0_x1.csv.gz'
    try:
        df = pd.read_csv(x0_x1_fn, sep='\t')
        pred_avg_elongRate = ((df['x1'] - df['x0'])/ time).mean()
        return pred_avg_elongRate
    except:
        return np.nan


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
    result_df = pd.DataFrame(columns = ['gene', 'iteration', 'max_coverage', 'rate_calculated', 'true_avg_h', 'pred_avg_h'])
    for i in range(1,num_iterations+1):
        for gene in gene_list:
            folder = f'{large_folder}/iter_{i}/{gene}'
            max_coverage = calculate_max_coverage(folder, gene)
            true_avg_h = calculate_true_avg_elongRate(folder, gene) # kb/min
            rate_calculated = os.path.isfile(f'{folder}/pred_h.csv.gz')
            pred_avg_h = calculate_pred_avg_elongRate(folder, gene, time = time)
            result_df.loc[result_df.shape[0]] = [gene, i, max_coverage, rate_calculated, true_avg_h, pred_avg_h]
    result_df.to_csv(f'{large_folder}/max_coverage.csv.gz', index=False, sep = '\t', compression='gzip')
    print('Done')
