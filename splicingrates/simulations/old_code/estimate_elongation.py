import numpy as np
SEED= 9999
np.random.seed(SEED)
import matplotlib.pyplot as plt
import pandas as pd
from regression import piecewise_linear_regression as pwlr
import os

def calculate_culmulative_coverage(coverage_df, startT_idx, endT_idx, gapN=1):
    """
    Given the coverage_df, clean the coverage_df to make sure that the coverage at each timepoint is the culmulative coverage
    :param coverage_df: time-tagged coverage_df
    :param gapN: every gapN rows in the coverage_df, we will include a row to include into the piece-wise linear regression
    :return:
    """
    # Assuming df is your DataFrame
    df_cumulative = coverage_df.drop('position', axis=1).cumsum(axis=1) # do not change the location of this line of code
    df_cumulative['position'] = coverage_df['position']
    df_cumulative = df_cumulative[list(range(startT_idx, endT_idx+1)) + ['position']]  # only select the columns that we need
    # first, filter out the trailing rows (positions) where BOTH of the timepoints have zero coverage
    # last rows with non-zero coverage at time startT_idx
    last_non_zero_index_0 = df_cumulative[startT_idx][df_cumulative[startT_idx] != 0].last_valid_index()
    # last rows with non-zero coverage at time endT_idx
    last_non_zero_index_1 = df_cumulative[endT_idx][df_cumulative[endT_idx] != 0].last_valid_index()
    # then take the rows in coverage where both of the timepoints have non-zero coverage
    df_cumulative = df_cumulative.loc[:max(last_non_zero_index_0, last_non_zero_index_1)]
    # only select the gapN rows in the coverage_df
    df_cumulative = df_cumulative.iloc[::gapN]
    return df_cumulative

def calculate_x0_x1_from_regression(coverage_df, startT_idx, px0, py0, px1, py1, close0_thres=0.5):
    # first, to get x0, we care about genoimc position that do not include the trailing zeros of coverage (because those correpsond to positions that do not have any transcripts at t=0)
    last_non_zero_index_x0 = coverage_df[startT_idx][coverage_df[startT_idx] > close0_thres].last_valid_index()
    x0 = (coverage_df['position'].values)[:last_non_zero_index_x0+1]
    # then, to get x1, we first project y0 from x0. Then, given y0, we calcualte x1 by drawing a horizontal line from (x0,y0) to the piece-wise linear regression of the coverage at t=1
    y0 = pwlr.x_to_y_array(x0, px0, py0)
    x1 = pwlr.y_to_x_array(y0, px1, py1)
     # note that due to the nature of piece-wise linear regression, there can be cases where x1 has trailing nan because the corresponding y0 has values lower than the lowest y-coordinate in py1
    # we need to remove those trailing nan
    last_non_nan_index_x1 = np.where(~np.isnan(x1))[0][-1] # find the last index in x1 where the value is not nans
    last_idx_to_pick = min(last_non_zero_index_x0, last_non_nan_index_x1)
    x0 = x0[:last_idx_to_pick+1]
    y0 = y0[:last_idx_to_pick+1]
    x1 = x1[:last_idx_to_pick+1]
    # that means, for each position on the genome, at startT_idx, the coverage is y0, and at endT_idx, the same transcript that was at position x at startT_idx is now at position x1 at endT_idx
    return x0, x1, y0

def piecewise_linearRegression_no_trailing_zeroes(X, Y, endpoints = None, max_segments:int=15):
    '''
    We will first get rid of the trailing zeroes in the coverage data, and then fit the piece-wise linear regression
    Assumption: Y is the culmulative coverage at each position, so the step of calculating culmulative coverage has already been done
    :param X: position along the gene
    :param Y: coverage at each position
    :param max_segments: max number of segments to fit in piecewise linear regression
    :return:
    '''
    # first, filter out the trailing rows (positions) where BOTH of the timepoints have zero coverage
    last_non_zero_index = np.where(Y != 0)[0][-1]
    X = X[:last_non_zero_index+1]
    Y = Y[:last_non_zero_index+1]
    px, py = pwlr.piecewise_linearRegression(X, Y, fixed_break=endpoints, maxcount=max_segments)
    return px, py

def estimate_endpoints_acrossTime(coverage_df, startT_idx, endT_idx, endpoints = None, max_segments:int=15, gapN:int=10):
    """
    Given the TIME-TAGGED coverage dataframe (index: genomic position, columns: timepoints), data is the time-tagged read covergage, the goal is to calculate the elongation speed h for the transcripts
    :param coverage_df: rows correspond to different positions on the genome, and columns correspond to different timepoints. entries: time-tagged coverage
    :param gapN: every gapN rows in the coverage_df, we will include a row to include into the piece-wise linear regression
    :return:
     - px0, py0: list of points that connects segments of the piece-wise linear regression of read coverage at the start time points
     - px1, py1: list of points that connects segments of the piece-wise linear regression of read coverage at the end time points
    - x0: a list of x-coordinates that correspond to the genomic position at start time points
    - x1: a list of x-coordinates that correspond to the genomic position at end time points
    x1 and x0 elements correspond to a 1-1 mappings of positions (if a transcript is at position x0 at startT_idx, it is at position x1 at endT_idx)
    """
    # first clean the coverage_df to (1) get rid of trailing zeros, and (2) turn the time-tagged coverage to the culmulative coverage
    coverage_df = calculate_culmulative_coverage(coverage_df, startT_idx, endT_idx, gapN=gapN)
    # first fit two piece-wise linear regressions to the coverage data for both timepoints
    # here, x is the genomic position, and y is the coverage at time t
    px0, py0 = piecewise_linearRegression_no_trailing_zeroes(coverage_df['position'].values, coverage_df[startT_idx].values, endpoints=endpoints, max_segments=max_segments)
    px1, py1 = piecewise_linearRegression_no_trailing_zeroes(coverage_df['position'].values, coverage_df[endT_idx].values, endpoints=endpoints, max_segments=max_segments)
    # TODO: come up with a better solution than to have to do minor_fix_py
    # For each position on the genome, find the expected coverage at time startT_idx
    x0, x1, y0 = calculate_x0_x1_from_regression(coverage_df, startT_idx, px0, py0, px1, py1)
    # that means, for each position on the genome, at startT_idx, the coverage is y0, and at endT_idx, the same transcript that was at position x at startT_idx is now at position x1 at endT_idx
    return px0, py0, px1, py1, x0, x1, y0

def findA(x0, x1, endpoints):
    n = len(x0)
    m = len(endpoints)-1 # endpoints include the first position in the gene, so the number of features is len(endpoints)-1, with the last feature corresponding to the run through region
    A = np.zeros((n,m)) # this is the coefficient matrix that we will construct. Each entry corresponds to the length of the portion within the segment between x0 and x1 that falls within the feature of endpoints
    for sample_idx in range(n):
        this_x0 = x0[sample_idx]
        this_x1 = x1[sample_idx]
        for i in range(m):
            if this_x0 < endpoints[i]:
                break
            if this_x0 > endpoints[i+1]: # this entry starts after the end of this feature
                continue
            if this_x1 < endpoints[i]: # this entry ends before the start of this feature
                break # no need to continue since A is initally filled with zeros
            if this_x0 >= endpoints[i] and this_x0 < endpoints[i+1]: # this entry starts within this feature
                if this_x1 > endpoints[i+1]: # this entry ends after the end of this feature
                    A[sample_idx,i] = endpoints[i+1] - this_x0
                    this_x0 = endpoints[i+1]
                    continue # go to the next feature
                else: # this entry ends within this feature
                    A[sample_idx,i] = this_x1-this_x0
                    break # no need to continue to the following features since A is initally filled with zeros
    return A

def find_optimal_trimN(x0,x1,trim_N):
    '''
    Given the vectors x0 and x1, find the optimal number of datapoints to trim from the start and end of the vectors
    Here, optimal is very simply. If x0 and x1 has trailling nans, find the number of nan from each side
    If the number of nan is smaller than the user-specified trim_N, then return trim_N, else return the number of nan
    Why do we need trim_N? Because the piece-wise linear regression can sometimes produce trailing nans in x0 and x1, because the x values are outside the range of the endpoints of the piece-wise linear regression
    :param x0: a vector of x-coordinates, the coordinate values of transcripts at startT_idx
    :param x1: a vector of x-coordinates, the coordinate values of the corresponding transcripts at endT_idx
    :param trim_N: user-specified number of datapoints to trim from the start and end of the vectors
    :return:
    '''
    # find the first and last non-nan index in x0 and x1
    first_non_nan_index_x0 = np.where(~np.isnan(x0))[0][0]
    last_non_nan_index_x0 = np.where(~np.isnan(x0))[0][-1]
    first_non_nan_index_x1 = np.where(~np.isnan(x1))[0][0]
    last_non_nan_index_x1 = np.where(~np.isnan(x1))[0][-1]
    # find the number of nan from each side
    num_nan_start = max(first_non_nan_index_x0, first_non_nan_index_x1)
    num_nan_end = max(len(x0)-last_non_nan_index_x0-1, len(x1)-last_non_nan_index_x1-1)
    # return the optimal number of nan to trim, we like overtrimming than undertrimming
    return max(trim_N, num_nan_start, num_nan_end)

def calculate_x0_x1_from_regression(coverage_df, startT_idx, px0, py0, px1, py1):
    # first, to get x0, we care about genoimc position that do not include the trailing zeros of coverage (because those correpsond to positions that do not have any transcripts at t=0)
    last_non_zero_index_x0 = coverage_df[startT_idx][coverage_df[startT_idx] != 0].last_valid_index()
    x0 = (coverage_df['position'].values)[:int(last_non_zero_index_x0+1)]
    # then, to get x1, we first project y0 from x0. Then, given y0, we calcualte x1 by drawing a horizontal line from (x0,y0) to the piece-wise linear regression of the coverage at t=1

    y0 = pwlr.x_to_y_array(x0, px0, py0)
    x1 = pwlr.y_to_x_array(y0, px1, py1)
     # note that due to the nature of piece-wise linear regression, there can be cases where x1 has trailing nan because the corresponding y0 has values lower than the lowest y-coordinate in py1
    # we need to remove those trailing nan
    last_non_nan_index_x1 = np.where(~np.isnan(x1))[0][-1] # find the last index in x1 where the value is not nans
    x0 = x0[:last_non_nan_index_x1+1]
    y0 = y0[:last_non_nan_index_x1+1]
    x1 = x1[:last_non_nan_index_x1+1]
    # that means, for each position on the genome, at startT_idx, the coverage is y0, and at endT_idx, the same transcript that was at position x at startT_idx is now at position x1 at endT_idx
    return x0, x1, y0

def subsample_x0_x1_for_leastSquare(x0, x1, frac=0.3, SEED=9999, trim_endGene=False, gtf_df=None, num_iters=100):
    """
    Given the vectors x0 and x1 which covers all the bases on the gene (where they are at t=0 and where they are at t=1)
    We can theoretically use all of them to calculate the elongation rates
    However, we can try to subsample them, and calculate the rates multiple times.
    This can have the effect of reducing the noise in the elongation rate estimation
    And potentially, we can use the subsampled elongation rates to calculate the mean and standard deviation of the elongation rates
    :param x0: a vector of x-coordinates, the coordinate values of transcripts at startT_idx
    :param x1: a vector of x-coordinates, the coordinate values of the corresponding transcripts at endT_idx
    :param endpoints: a list of x-coordinates that correspond to the endpoints of the features in the gene
    :param trim_N: number of datapoints to trim from the start and end of the vectors. This maybe needed because when we calculate x1 based on the piece-wise linear regression, we can have trailing nans because those corresponds to regions not covered by the piece-wise linear regression.
    :param frac: the fraction of the data to subsample
    :param SEED: random seed for reproducibility
    :param trim_endGene: whether to trim AWAY the data of positions that are beyond the last feature in the gene at t=0, since they do not have any values for elongation rates of interest
    :param gtf_df: the gtf dataframe. This is needed when trim_endGene is True
    :param num_iters: number of iterations to subsample the data
    :return: x0_, x1_: the trimmed and subsampled x0 and x1. size: (num_iters, m) where m is the number of subsampled data points
    """
    # then, if users want to trim away positions started beyong the end odf the gene, we will do that
    if trim_endGene:
        # get the last positions of features in the gtf_df
        geneEnd = gtf_df['end'].max()
        x0_ = x0[x0 <= geneEnd]
        x1_ = x1[x0 <= geneEnd]  # the indices are chosen such that we only care about positions that start BEFORE the gene end
    # then, subsample the vectors. We will repeat this process num_iters times
    np.random.seed(SEED)
    N = len(x0)
    m = np.max([int(N*frac), gtf_df.shape[0]+1])  # we want to subsample at least the number of features in the gene, for later regression A*h_inv = b to be identifiable
    subsampling_result = np.random.choice(np.arange(N), (num_iters, m), replace=True)
    x0_ = (x0[subsampling_result]).reshape(num_iters, m)
    x1_ = (x1[subsampling_result]).reshape(num_iters, m)
    return x0_,x1_

def get_endpoints_from_gtf(gtf_df, convert_to_KB=True):
    '''
    This function will create a list of endpoints of features from gtf_df. The output is needed to calculate the coefficient matrix A
    :param gtf_df: df of the gene annotation file
    :param convert_to_KB: True if we want to convert genomic position to KB. This will help the calculation of h to be more numerically stable. Output will be in KB/min
    :return: endpoints
    '''
    gene_length = gtf_df['end'].max()
    # endpoints = np.arange(0, gene_length+1, 200)  # the endpoints of the features in the gene
    # get the endpoints of features in the gtf_df
    # get the endpoints of features in the gtf_df
    endpoints = gtf_df[['start', 'end']].values
    # assert that the end of row i is the same as the start of row i+1
    assert np.all(endpoints[:-1, 1] == endpoints[1:, 0]), 'the endpoints of features in gtf_df should be continuous. Here, it is not, please check the gtf_df'
    endpoints = endpoints[:,0] #np.concatenate((endpoints[:, 0], endpoints[-1, 1]))  # the last endpoint of the last feature is the end of the gene
    endpoints = np.append(endpoints, np.inf) # add a very large number to the end of the endpoints list to make sure that the last segment (beyond the end of all features) is included in the calculation
    if convert_to_KB:
        endpoints = endpoints/1000
    return endpoints

def calculate_h_one_round(x0, x1, endpoints, time):
    """
    Given the vectors x0 and x1, and the endpoints of the features in the gene, calculate the elongation rate h
    :param x0: a vector of x-coordinates, the coordinate values of transcripts at startT_idx
    :param x1: a vector of x-coordinates, the coordinate values of the corresponding transcripts at endT_idx
    :param endpoints: a list of x-coordinates that correspond to the endpoints of the features in the gene
    :param time: actual time (in minutes) from startT_idx to endT_idx
    :return: h
    """
    n = len(x0) # number of datapoints that we have to estimate elongation rate
    # n: number of samples (positions along the gene) and m: number of features/bins in the gene
    # based on x0,x1 and endpoints, find the coefficient matrix A (size n*m)
    A = findA(x0, x1, endpoints)
    print('A:', A)
    b = np.ones(n) * time
    # Computing the least squares solution
    from scipy.optimize import lsq_linear
    res = lsq_linear(A, b, bounds=(0.01, np.inf))  # A*h_inv = b, with A(n,m), h_inv(m,), b(n,)
    h = 1 / res.x
    return h

def convert_gene_to_KB(coverage_df, convert_to_KB=True):
    '''
    Given the coverage_df, convert the genomic position to KB
    :param coverage_df:
    :return: column 'position' in coverage_df is converted to KB
    '''
    if convert_to_KB:
        print('We will conver the genomic positions in terms of kilo-bases')
        print('Output will be in KB/min')
        coverage_df['position'] = coverage_df['position']/1000
    return coverage_df

def clean_edgeEffect_region_from_coverage(coverage_df, trim_N=300):
    '''
    Given the coverage_df, get rid of the first and last trimN positions, they overlap with regions that we believe are affected by edge effects
    :param coverage_df:
    :param trim_N:
    :return:
    '''
    # get rid of rows where the coverage is zero at all timepoints
    position_df = coverage_df['position'].copy()
    coverage_df = coverage_df.drop('position', axis=1)
    coverage_df = coverage_df.loc[(coverage_df != 0).any(axis=1)]
    if trim_N!=0:
        coverage_df = coverage_df.iloc[trim_N:-trim_N]
    coverage_df = coverage_df.merge(position_df, left_index=True, right_index=True)
    return coverage_df

def calculate_h_from_coverage(coverage_df, gtf_df, startT_idx=0, endT_idx=1, time=5, max_segments=15, trim_endGene=False, trim_N=300, subsample_frac=0.3, SEED=9999, nIter=100, convert_to_KB=False, gapN=10):
    """
    Given the TIME-TAGGED coverage dataframe (index: genomic position, columns: timepoints), data is the time-tagged read covergage, the goal is to calculate the elongation speed h for the transcripts
    :param coverage_df: time-tagged coverage_df
    :param gtf_df: gtf_df showing the features of the gene for each feature we aim to find the elongation speed
    :param startT_idx: index of the start timepoint in the coverage_df that we will collect data for elongation rate estimation
    :param endT_idx: index of the end timepoint in the coverage_df that we will collect data for elongation rate estimation
    :param time: actual time (in minutes) from startT_idx to endT_idx
    :param max_segments: parameter for pwlr.piecewise_linearRegression function representing the maximum number of segments to fit the shape of the read coverage
    :param trim_N: number of datapoints to trim from the start and end of the coverage_df when we try to calculate h from the regressed values of x0 and x1
    :param trim_endGene: whether to trim AWAY the data of positions that are beyond the last feature in the gene at t=0, since they do not have any values for elongation rates of interest
    :param subsample_frac: the fraction of the data of x0, x1 to subsample in each iteration (here, subsamplilng WITH replacement, because it makes for an elegant implementation)
    :param convert_to_KB: True if we want to convert genomic position to KB. This will help the calculation of h to be more numerically stable. Output will be in KB/min
    :return:
    """
    coverage_df = convert_gene_to_KB(coverage_df, convert_to_KB)
    # now filter the read coverage: get rid of the first and last trimN positions, they overlap with regions that we believe are affected by edge effects
    coverage_df = clean_edgeEffect_region_from_coverage(coverage_df, trim_N)
    print('Start calculating elongation rate')
    # first, calculate the estimated endpoints of transcripts from startT_idx to endT_idx, based on the time-tagged coverage
    assert startT_idx < endT_idx, 'startT_idx should be smaller than endT_idx'
    x0 = np.array([])
    x1 = np.array([])
    # find the endpoints of features in the gtf_df
    endpoints = get_endpoints_from_gtf(gtf_df, convert_to_KB)
    print('endpoints:', endpoints)
    for i in range(startT_idx, endT_idx): # we can use the data for multiple timepoints to estimate the endpoints of the transcripts. It is simply a matter of concatenating the x0 and x1 estimated from multiple timepoints.
        px0, py0, px1, py1, x0_, x1_, _ = estimate_endpoints_acrossTime(coverage_df, i, i+1, endpoints=np.arange(0, 5, 0.2), max_segments= max_segments, gapN=gapN)
        x0 = np.concatenate((x0, x0_))
        x1 = np.concatenate((x1, x1_))
    # x0, x1 are the ABSOLUTE genomic positions (starting from gene start) of the transcripts at startT_idx and endT_idx, respectively, outputted by the piece-wise linear regression.
    print('Done getting the piece-wise linear regression')
    # we want to sample the values of x0 and x1 for multiple iterations of calcualtion of h
    x0_, x1_ = subsample_x0_x1_for_leastSquare(x0, x1, subsample_frac, SEED, trim_endGene, gtf_df, nIter)
    print('Done subsampling x0 and x1')
    # calculate h for each subsampled x0 and x1
    h = np.zeros((nIter, len(endpoints)-1))  # bc endpoints includes TSS at the begining of the gene, so the number of features is len(endpoints)-1
    for i in range(nIter):
        h[i] = calculate_h_one_round(x0_[i], x1_[i], endpoints, time)
    print('Done solving the elongation rate problem')
    rows_without_nan = ~np.any(np.isnan(h), axis=1)
    # Select only those rows
    h = h[rows_without_nan,:]
    print('Out of {num_iters} iterations, {num_valid} iterations produced valid (no nan at all) longation rates'.format(num_iters=nIter, num_valid=h.shape[0]))
    h_mean = np.mean(h, axis=0)  # average the elongation rates for each feature across iterations
    return h, h_mean


