'''
Given a gtf file of gene features, check if the genes are continuous, meaning that the exons and introns are all continuously placed next to each other.
Conclusion: all genes are continuous
'''
import pandas as pd
import numpy as np
import os
import sys
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Check continuous genes")
    parser.add_argument('--gtf_fn', type=str, help='input gene features')
    args = parser.parse_args()
    df = pd.read_csv(args.gtf_fn, sep='\t', header=0, index_col=None)
    # check that the difference between the end of one feature and the start of the next feature is 1
    # if it is not, then the gene is not continuous
    df['prev_end'] = df['end'].shift(1)
    df['diff'] = df['start'] - df['prev_end']
    # check that the diff in the df is all 1 except for 1 nan
    assert df['diff'].isnull().sum() == 1
    assert (df['diff'] == 1).sum() == df.shape[0] - 1, 'Did not see that all the difference between this start and previous end is 1. Check the gene features.'
    print('This gene is continuous.')


