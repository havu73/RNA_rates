import os
import numpy as np
from transcription import elongation_calculation as elong
import estimate_elongation as estElong
from regression import piecewise_linear_regression as pwlr
import pandas as pd
ONE_KB=1000
SIM_FEAT_LEN = 5000 # length of one feature in the simulation
SEED = 9999
np.random.seed(SEED)


def trial_first_position_decrease_coverage(current_cov, culmu_cov_afterI_df, endT_idx=2):
    lower_positions = culmu_cov_afterI_df[culmu_cov_afterI_df[endT_idx] < current_cov]['position']
    if not lower_positions.empty:
        return lower_positions.iloc[0]
    return current_cov

def calculate_splice_given_coverage_df(coverage_df, intron_end, gtf_df, endT_idx=2):
    '''
    :param coverage_df:
    :param intron_end:
    :param gtf_df:
    :return:
    '''
    # get the coverage before and after the intron end
    coverage_afterI_df = coverage_df[coverage_df['position'] >= intron_end]
    culmu_cov_afterI_df = estElong.calculate_culmulative_coverage(coverage_afterI_df, startT_idx=0, endT_idx=endT_idx, gapN=1)
    endpoints = estElong.get_endpoints_from_gtf(gtf_df, convert_to_KB=False)
    px, py = estElong.piecewise_linearRegression_no_trailing_zeroes(culmu_cov_afterI_df['position'].values, culmu_cov_afterI_df[endT_idx].values, endpoints=endpoints, max_segments=15)
    # get the coverage before the intron end
    coverage_beforeI_row = coverage_df.loc[intron_end-1]
    # create a df where the rows correspond to each of the coverage point right after the intronEnd
    culm_right_afterI_cov = culmu_cov_afterI_df.loc[intron_end]  # 0: 110, 1: 230, 2: 340
    trans_afterI_df = pd.DataFrame({'coverage': np.arange(culm_right_afterI_cov[endT_idx]),
                                    'tagged_time': np.zeros(culm_right_afterI_cov[endT_idx])})
    for time_idx in range(1, endT_idx+1):
        start_trans_idx = culm_right_afterI_cov[time_idx-1]
        end_trans_idx = culm_right_afterI_cov[time_idx]
        trans_afterI_df.loc[start_trans_idx:end_trans_idx, 'tagged_time'] = time_idx
    trans_afterI_df['endpoint'] = trans_afterI_df['coverage'].apply(lambda ys: pwlr.y_to_x_array(np.array([ys]), px, py)) #trans_afterI_df['coverage'].apply(lambda y: trial_first_position_decrease_coverage(y, culmu_cov_afterI_df))
    #trans_afterI_df['coverage'].apply(lambda ys: pwlr.y_to_x_array(np.array([ys]), px, py))
    # if endpoint is nan, it means that the transcript is very close to  the end of the intron and its endpoint is probably at exactly the end of the intron --> replace na with the minimum endpoint value that is not nan
    min_endpoint = np.nanmin(trans_afterI_df['endpoint'].values)
    trans_afterI_df['endpoint'] = trans_afterI_df['endpoint'].fillna(min_endpoint)
    # now, given the true elongation rate, calculate the time it takes to traverse from the intron end to the endpiont of the transcript
    trans_afterI_df['time_since_endI'] = trans_afterI_df['endpoint'].apply(lambda x: elong.time_to_elongate(intron_end, x, gtf_df))
    # now, calculate h
    h_dict = find_h(trans_afterI_df, coverage_beforeI_row, endT_idx=endT_idx)
    return h_dict # each entry is a prediction of h based on slightly different ways of calculating it



def objective_function(h, t_i_list, n_b_list):
    '''

    :param h: splice half-life that we are trying to solve
    :param t_i_list: list of t_i values. Each element is an array of t_i values tagged to a certain time point, where each t_i is estimated time for the transcript to traverse from the intron end to the transcript endpoint at the end of the experiment.
    :param n_b_list: list of n_b values. Each element is the sum of the number of UNSPLICED transcripts tagged at each timepoint (estimated from the coverage data of positions right before the intron end)
    :return:
    '''
    assert len(t_i_list) == len(n_b_list), 'Caclulation error: the length of t_i_list and n_b_list should be the same'
    # Calculate the sum of 2^(-t_i / h) for i from 1 to n_a  -> the probability of a transcript being unspliced given its endpoint at t_i
    to_minimize = 0
    for i, (t_i, n_b) in enumerate(zip(t_i_list, n_b_list)):
        S = np.sum(2 ** (-t_i / h))
        to_minimize += ((S - n_b)**2)/len(t_i)  # mean squared error, basically
    # Return the difference between the sum and n_b
    return to_minimize

def objective_rootFinding(h, t_i_list, n_b_list):
    assert len(t_i_list) == len(n_b_list), 'Caclulation error: the length of t_i_list and n_b_list should be the same'
    to_be_zero = 0
    for i, (t_i, n_b) in enumerate(zip(t_i_list, n_b_list)):
        S = np.sum(2 ** (-t_i / h))
        to_be_zero += (S - n_b)/len(t_i)  # weighted by the number of data points
    return to_be_zero

def find_root(t_i_list, n_b_list):
    from scipy.optimize import root_scalar
    try:
        sol = root_scalar(objective_rootFinding, args=(t_i_list, n_b_list), bracket=[1, 20], method='brentq')
    except:
        print("The root-finding process did not converge.")
        return np.nan
    return sol.root


def optimize_h(t_i_list, n_b_list):
    '''
    :param t_i_list: list of t_i values. Each element is an array of t_i values tagged to a certain time point, where each t_i is estimated time for the transcript to traverse from the intron end to the transcript endpoint at the end of the experiment.
    :param n_b_list: list of n_b values. Each element is the sum of the number of UNSPLICED transcripts tagged at each timepoint (estimated from the coverage data of positions right before the intron end)
    :return:
    '''
    from scipy.optimize import minimize
    # Use a root-finding method to solve for h
    bounds = [(1, 50)]
    initial_guess = np.arange(bounds[0][0], bounds[0][1], 3)
    results = []
    for guess in initial_guess:
        result = minimize(objective_function, guess, args=(t_i_list, n_b_list), bounds=bounds)
        if result.success:
            results.append(result)
    best_result = min(results, key=lambda res: res.fun)
    return best_result.x[0]

def find_h(trans_afterI_df, coverage_beforeI_row, endT_idx=2):
    t_i_list = []  # each element corresponds to a time point
    n_b_list = []
    h_dict = {} # list of h values that we will calculate based on different ways
    for time_idx in range(endT_idx+1):
        t_i = trans_afterI_df[trans_afterI_df['tagged_time']==time_idx]['time_since_endI'].values
        n_b = coverage_beforeI_row[time_idx]
        h = optimize_h([t_i], [n_b])
        h_dict[f'h_{time_idx}'] = h
        t_i_list.append(t_i)
        n_b_list.append(n_b)
    h_dict['h_all'] = optimize_h([trans_afterI_df['time_since_endI'].values],
                                 [np.sum(coverage_beforeI_row[list(range(endT_idx+1))], axis=0)])
    # Use a root-finding method to solve for h
    h_dict[f'h_comb'] = optimize_h(t_i_list, n_b_list)
    t_i_list.append(trans_afterI_df['time_since_endI'].values)
    n_b_list.append(np.sum(coverage_beforeI_row[list(range(endT_idx+1))], axis=0))
    h_dict['h_comb_and_all'] = optimize_h(t_i_list, n_b_list)
    return h_dict