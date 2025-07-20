import pandas as pd
import numpy as np
import argparse
import helper
import torch
import os
import itertools
from torch.utils.data import Dataset, DataLoader
import sys
from run_solvers import findA, solve
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from train_loop import SimpleSolver, BayesianNormalSolver, BayesianRBFSolver, BayesianLogNormalSolver
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from transcription.elongation_calculation import calculate_enlongated_endsite
from estimates.utils import merge_intervals

# first, simulate data of A.
def simulate_A(n, elongf_df, label_time, G = 15, h_bins = 0.2, eps_mu=0, eps_var = 0.1):
    '''
    Simulate the matrix A that represents the gene with n samples and m features
    :param n: number of samples
    :param distance: distance between the start and end of the transcript between elongation period
    :param G: length of the gene (kb)
    :param h_bins: bin size of each feature in the gene (kb)
    :param eps_mu: mean of the error term
    :param eps_var: variance of the error term
    :return: A
    '''
    ONE_KB = 1000
    # first, sample n random start points from 0 to G
    x0 = np.random.uniform(0, G, n)
    # beacuse the function calculate_enlongated_endsite operates on the location of start and end sites at bp-level, we need to convert the kb to bp and then convert back from bp to kb
    vectorized_function = np.vectorize(lambda x: calculate_enlongated_endsite(prev_stop=x*ONE_KB, elongf_df= elongf_df, time_since_prev=label_time))
    x1 = vectorized_function(x0)/ONE_KB
    # for each endpoint, we can add a small error term epsilon
    epsilon = np.random.normal(eps_mu, np.sqrt(eps_var), n)
    x1 += epsilon
    # get the endpoints based on h_bins and G
    endpoints = np.arange(0, G, h_bins)
    endpoints = np.append(endpoints, np.inf)
    # A: overlap matrix between each segment and each feature
    A = findA(x0, x1, endpoints)
    return A, endpoints

def calculate_avg_h(A, b):
    '''
    Given A and b, calculate the average elongation rate across the gene
    '''
    # sum each row of A, and divide by b, then average across all rows
    return np.mean(np.sum(A, axis=1) / b)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="This code will take in different possible values of parameters, and then create different simulate of input data for the solvers to deal with.")
    parser.add_argument('--N', required=True, type=int, help='Number of rows in the A matrix that we will simulate')
    parser.add_argument('--G', required=True, type=float, help='Different possible values of the length of the gene')
    parser.add_argument('--elongf_fn', required=True, type=str, help='file that contains the elongation rates')
    parser.add_argument('--label_time', required=True, type=int, help='length of labeling time (mins). this is the 3 * label_time per label in the kinetic barcoding experiment')
    parser.add_argument('--h_bin', required=True, type=float, help='length of each bin for which we will try to solve for the elongation rate')
    parser.add_argument('--seed', required=False, type=int, default=9999, help='Different possible values of the seed')
    parser.add_argument('--output_fn', required=True, type=str, help='Output of the analysis')
    parser.add_argument('--lambda_smooth', required=False, type=float, default=1, help='lambda for the smoothness of the elongation rate')
    args, unknown = parser.parse_known_args()
    helper.create_folder_for_file(args.output_fn)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    elongf_df = pd.read_csv(args.elongf_fn, header = 0, index_col =None, sep = '\t')
    A, breaks = simulate_A(args.N, elongf_df, label_time=args.label_time, G = args.G, h_bins=args.h_bin)
    b = np.ones(args.N) * args.label_time
    print('Done with getting simulation data')
    result_df = pd.DataFrame()
    result_df['start'] = breaks[:-1]
    result_df['end'] = breaks[1:]
    result_df.loc[result_df.shape[0]-1, 'end'] = args.G
    init_h = calculate_avg_h(A, b)
    result_df = solve(A, b, breaks, result_df, init_h, use_simpleSolver_init=True, lambda_smooth=args.lambda_smooth)  # start, end, simpleSolver, bayesLinearSolver, bayesRBFSolver, logNormalSolver
    # then, we have to combine the predicted elongation rates with the true elongation rates into one dataframe, for later investigation
    # beacuse the code to merge intervals between two dfs assumes the start and end in the two dfs are on the same units (bp or kb), we need to convert for the elongf_df to be on the same units as the result_df
    ONE_KB=1000
    elongf_df['start'] = elongf_df['start']/ONE_KB
    elongf_df['end'] = elongf_df['end']/ONE_KB
    elongf_df['length'] = elongf_df['length'] / ONE_KB
    methods = ['simpleSmooth']  #, 'simpleSolver', 'bayesLinearSolver', 'bayesRBFSolver', 'logNormalSolver']
    result_df = merge_intervals(result_df, elongf_df, value1=methods, value2=['txrate'])
    # save the results
    result_df.to_csv(args.output_fn, index=False, header=True, sep='\t', compression = 'gzip')
    print(f"Output saved to {args.output_fn}")


