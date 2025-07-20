import os
import numpy as np
import seaborn as sns
# add the path of the parent directory to the path
import sys

sys.path.append('../')
import visualize_simulations as viz
import matplotlib.pyplot as plt
plt.style.use('default')
plt.rcParams['figure.facecolor'] = 'white'
from estimates.elongation import estElong, find_gap
import helper
import pandas as pd
ONE_KB=1000
SIM_FEAT_LEN = 5000 # length of one feature in the simulation
SEED = 9999
np.random.seed(SEED)
import pandas as pd
import numpy as np
import argparse


def read_one_file(fn, timepoint=None):
    df = pd.read_csv(fn, sep='\t', header=None)
    df.columns = ['chrom', 'start', 'end', 'gene', 'dot', 'strand', 'coord', 'coverage']
    if timepoint is not None:
        df.rename(columns={'coverage':  timepoint}, inplace=True)
    strand = df['strand'].unique()[0]
    return df, strand

def merge_dfs(df_list):
    '''
    Given all the df, merge them by the coord column
    :param df_list:
    :return:
    '''
    df = df_list[0]
    for i in range(1, len(df_list)):
        df = pd.merge(df, df_list[i][['coord', i]], on='coord', how='outer')
    return df


def transform_jesse_gtf_df(df, strand="+"):
    df = df[['nucleotide_coord', 'rate_for_this_nt']].copy()
    # Step 1: Rename the columns
    df = df.rename(columns={'nucleotide_coord': 'end', 'rate_for_this_nt': 'true_h'})
    df['true_h'] = df['true_h'] / 1000 # bc Jesse's rates are in nt/s
    # Step 2: Create a start column
    df['start'] = df['end'] - 1
    df['diff'] = df['true_h'].diff()
    df['diff'] = df['diff'].fillna(0)
    df['group'] = (df['diff'] != 0).cumsum()
    output_df = df.groupby('group').agg({'start': 'first', 'end': 'last', 'true_h': 'first'}).reset_index(drop=True)
    output_df['length'] = output_df['end'] - output_df['start']
    output_df['feature'] = 'gene'
    # if strand == '-': then reverse the order of the rows for the true_h column, since we also rever the coverage of the gene (not the coordinate)
    if strand == '-':
        output_df['true_h'] = output_df['true_h'].values[::-1]
    # Add the RTR line
    last_h = output_df.loc[output_df.shape[0]-1, 'true_h']
    RTR_line = {'start': output_df['start'].max(), 'end': output_df['end'].max() + 1, 'true_h': last_h, 'length': 1, 'feature': 'RTR'}
    output_df.loc[output_df.shape[0]] = RTR_line
    return output_df



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Estimate elongation rates")
    parser.add_argument('--gene_name', type=str, required=True, help="Gene name")
    parser.add_argument('--data_folder', type=str, required=True, help="Folder containing the data")
    args = parser.parse_args()
    fn1 = f'{args.data_folder}/{args.gene_name}/{args.gene_name}_nt_coverage_0-5.bed.gz'
    fn2 = f'{args.data_folder}/{args.gene_name}/{args.gene_name}_nt_coverage_5-10.bed.gz'
    fn3 = f'{args.data_folder}/{args.gene_name}/{args.gene_name}_nt_coverage_10-15.bed.gz'
    df1, strand = read_one_file(fn1, 0)
    df2, strand = read_one_file(fn2, 1)
    df3, strand = read_one_file(fn3, 2)
    df = merge_dfs([df1, df2, df3])
    if strand == '-':
        df[0] = df[0].values[::-1]
        df[1] = df[1].values[::-1]
        df[2] = df[2].values[::-1]
    gtf_fn = f'{args.data_folder}/{args.gene_name}/{args.gene_name}.gtf.gz'
    gtf_df = pd.read_csv(gtf_fn, header=0, index_col=None, sep='\t')
    clean_gtf_df = transform_jesse_gtf_df(gtf_df, strand)
    coverage_df = df[['coord', 0, 1, 2]].copy()
    coverage_df.rename({'coord': 'position'}, axis=1, inplace=True)
    output_folder=f'{args.data_folder}/{args.gene_name}'
    elong_estimator = estElong(coverage_df, clean_gtf_df, h_bin_bp=[100, 500, 1000, 2000], startT_idx=0, endT_idx=2,
                                    output_folder=output_folder)
    result_df = elong_estimator.estimate()
    print(result_df.head(30))
    save_fn = f'{output_folder}/regression_lines.png'
    elong_estimator.draw_regression_lines(save_fn)
    save_fn = f'{output_folder}/pred_h.csv.gz'
    elong_estimator.save_estimates(save_fn)

