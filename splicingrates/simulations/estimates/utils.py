import pandas as pd
import numpy as np
import torch

def drop_trailing_zeros(df):
    '''
    Given a df, we want to get rid of all the rows at the end where all the values are zero
    :param df:
    :return:
    '''
    mask = (df != 0).any(axis=1)  # Create a mask where rows with any non-zero values are True
    return df.iloc[:int(mask[::-1].idxmax() + 1)]  # Reverse mask, find first True from end, slice DataFrame

def merge_intervals(df1, df2, value1=['value1'], value2=['value2']):
    '''
    Given two dataframes with columns ['start', 'end', value\in['pred_h', 'true_h' or something like that]], merge the intervals and deduplicate the intervals
    Function was actually created by ChatGPT. It is NOT optimized for speed but it works for the case where df1 and df2 are not too large, bc it create a df that has nrows = len(df1) * len(df2)
    :param df1:
    :param df2:
    :return:
    '''
    # Step 1: Merge the dataframes on overlapping intervals
    merged_df = pd.merge(df1.assign(key=1), df2.assign(key=1), on='key', suffixes=('1', '2')).drop('key', axis=1)
    merged_df = merged_df[(merged_df['start1'] < merged_df['end2']) & (merged_df['end1'] > merged_df['start2'])]
    # Step 2: Expand the intervals to reflect all unique combinations
    merged_intervals = []
    epsilon = 1e-5  # Small value to handle floating point errors
    for _, row in merged_df.iterrows():
        start = max(row['start1'], row['start2'])
        end = min(row['end1'], row['end2'])
        if start+epsilon < end:  # sometimes the intervals are so close that they are considered the same
            insert_row = pd.Series([start, end], index=['start', 'end'])
            insert_row = pd.concat([insert_row, row[value1 + value2]])
            merged_intervals.append(insert_row)
    result_df = pd.DataFrame(merged_intervals, columns=['start', 'end'] + value1 + value2)
    # Step 3: Deduplicate the intervals
    result_df = result_df.drop_duplicates().sort_values(by=['start', 'end']).reset_index(drop=True)
    return result_df

def _filterOut_nan_numpy(x0, x1):
    mask0 = np.isnan(x0)
    mask1 = np.isnan(x1)
    mask = mask0 | mask1
    x0_filtered = x0[~mask]
    x1_filtered = x1[~mask]
    return x0_filtered, x1_filtered


def _filterOut_nan_torch(x0,x1):
    mask0 = torch.isnan(x0)
    mask1 = torch.isnan(x1)
    mask = mask0 | mask1
    x0_filtered = x0[~mask]
    x1_filtered = x1[~mask]
    return x0_filtered, x1_filtered


def filterOut_nan(x0, x1):
    if isinstance(x0, np.ndarray):
        return _filterOut_nan_numpy(x0, x1)
    elif isinstance(x0, torch.Tensor):
        return _filterOut_nan_torch(x0, x1)
    else:
        raise ValueError('filterOut_nan: The input should be either numpy array or torch tensor')


def checkA_zero_columns(A):
    '''
    Given the matrix A, filter out the last N column that are all zeroes
    :param A:
    :return:
    '''
    mask = np.all(A == 0, axis=0)
    # select up to the last non-zero column
    last_nonzero_idx = np.where(~mask)[0][-1]
    return A[:, :last_nonzero_idx + 1]


def findA(x0, x1, endpoints):
    '''
    Endpoints will have length m+1, where m is the number of bins for which we will find the elongation rates for.
    Endpoitns should have the first element equal to the start of the gene. The last element tends to be np.inf
    bc we may assume that the reads can run to beyond the end of the genes.
    :param x0:
    :param x1:
    :param endpoints:
    :return:
    '''
    assert len(x0)==len(x1), "x0 and x1 must have the same length"
    n = len(x0)
    m = len(endpoints) - 1  # endpoints include the first position in the gene, so the number of features is len(endpoints)-1, with the last feature corresponding to the run through region
    A = np.zeros((n, m))  # this is the coefficient matrix that we will construct. Each entry corresponds to the length of the portion within the segment between x0 and x1 that falls within the feature of endpoints
    for sample_idx in range(n):
        this_x0 = x0[sample_idx]
        this_x1 = x1[sample_idx]
        for i in range(m):
            if this_x0 < endpoints[i]:
                break
            if this_x0 > endpoints[i + 1]:  # this entry starts after the end of this feature
                continue
            if this_x1 < endpoints[i]:  # this entry ends before the start of this feature
                break  # no need to continue since A is initally filled with zeros
            if this_x0 >= endpoints[i] and this_x0 < endpoints[i + 1]:  # this entry starts within this feature
                if this_x1 > endpoints[i + 1]:  # this entry ends after the end of this feature
                    A[sample_idx, i] = endpoints[i + 1] - this_x0
                    this_x0 = endpoints[i + 1]
                    continue  # go to the next feature
                else:  # this entry ends within this feature
                    A[sample_idx, i] = this_x1 - this_x0
                    break  # no need to continue to the following features since A is initally filled with zeros
    return A
