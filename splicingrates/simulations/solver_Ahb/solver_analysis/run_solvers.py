import pandas as pd
import numpy as np
import argparse
import helper
import torch
import os
import itertools
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import train_loop
from train_loop import SimpleSolver, BayesianNormalSolver, BayesianRBFSolver, BayesianLogNormalSolver, SimpleSmoothSolver
from torch.utils.data import Dataset, DataLoader

# Example custom dataset
class CustomData(Dataset):
    def __init__(self, A, b):
        """
        Args:
            data (Tensor): Input data, e.g., features.
            targets (Tensor): Target data, e.g., labels.
        """
        self.data = torch.tensor(A).float()
        self.targets = torch.tensor(b).float()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_point = self.data[idx]
        target = self.targets[idx]
        return data_point, target

    def avg_pred_h(self):
        '''
        Given A and b, calculate the average pred_h across all samples
        :return:
        '''
        dist_traveled = torch.sum(self.data, dim = 1)
        pred_h = dist_traveled / self.targets
        return pred_h.mean(dim=0)




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

# first, simulate data of A.
def simulate_A(n, distance, G = 15, h_bins = 0.2, eps_mu=0, eps_var = 0.1):
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
    # first, sample n random start points from 0 to G
    x0 = np.random.uniform(0, G, n)
    x1 = x0 + distance
    # for each endpoints, we can add a small error term epsilon
    epsilon = np.random.normal(eps_mu, np.sqrt(eps_var), n)
    x1 += epsilon
    # get the endpoints based on h_bins and G
    endpoints = np.arange(0, G, h_bins)
    endpoints = np.append(endpoints, np.inf)
    # A: overlap matrix between each segment and each feature
    A = findA(x0, x1, endpoints)
    return A, endpoints

def run_one_method(d, method, breaks, true_h, dataloader, lambda_smooth = 1):
    nan_tensor = torch.tensor([np.nan]*d).cpu().numpy()
    if method == 'simpleSolver':
        Ssolver = SimpleSolver(d, init_h=true_h, dataloader=dataloader)  # solve for h such that A/h = b
        try:
            simple_h = Ssolver.solve()
            return simple_h.cpu().numpy()
        except:
            return nan_tensor
    if method == 'simpleSmooth':
        SSS = SimpleSmoothSolver(d, init_h=true_h, dataloader=dataloader, lambda_smooth = lambda_smooth)  # solve for h such that A/h = b, with some smoothness constraint
        try:
            avg_h = dataloader.dataset.avg_pred_h()
            simple_smooth_h = SSS.solve(avg_h=avg_h)
            return simple_smooth_h.cpu().numpy()
        except:
            return nan_tensor
    elif method == 'bayesLinearSolver':
        BNSolver = BayesianNormalSolver(d, init_h=true_h, dataloader=dataloader)  # solve for h such that A/h = b, with some prior distribution for h set by the solver
        try:
            bayes_linear_h = BNSolver.solve()
            return bayes_linear_h.cpu().numpy()
        except:
            return nan_tensor
    elif method == 'bayesRBFSolver':
        if len(breaks) == 2:  # the h_bin is too big compared to the length of the gene
            return nan_tensor
        assert (breaks[1] - breaks[0]) > 0 and (breaks[1] - breaks[0]) < torch.inf, 'The binsize for the elongation rates calculation should be positive and less than infirity'
        h_bins = breaks[1] - breaks[0]
        B_RBFSolver = BayesianRBFSolver(d, init_h=true_h, dataloader=dataloader, coords=breaks[:-1] + h_bins * 0.5)
        try:
            bayes_RBF_h = B_RBFSolver.solve()
            return bayes_RBF_h.cpu().numpy()
        except:
            return nan_tensor
    elif method == 'logNormalSolver':
        logNormalSolver = BayesianLogNormalSolver(d, init_h=true_h, dataloader=dataloader)
        try:
            logNormal_h = logNormalSolver.solve()
            return logNormal_h.cpu().numpy()
        except:
            return nan_tensor
    else:
        return nan_tensor

def solve(A, b, breaks,result_df, init_h=1, use_simpleSolver_init=False, lambda_smooth = 1):
    '''
    Solve for the elongation rates given A and b
    :param A: (n, m) where n is the number of samples and m is the number of features
    :param b: (n,) where n is the number of samples
    :param breaks: (m+1,) where m is the number of features
    :return: simpleSolver, bayesianLinearSolver, bayesianRBFsolver
    '''
    print('Solving for the elongation rates')
    print(A)
    # 1, Get the dataloader
    dataset = CustomData(A, b)
    batch_size = 2048
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    d = A.shape[1]
    # 2, Solve and save
    result_df['simpleSmooth'] = run_one_method(d, 'simpleSmooth', breaks, init_h, dataloader, lambda_smooth=lambda_smooth)
    print(result_df)
    # result_df['simpleSolver'] = run_one_method(d, 'simpleSolver', breaks, init_h, dataloader)
    # print(result_df)
    # if use_simpleSolver_init:  # if we want to use the output of simpleSolver as the initial values for the other solvers
    #     # if there are not negative entries in the result of simpleSolver, we can use it as the initial value for the other solvers
    #     if np.all(result_df['simpleSolver'] >= 0):
    #         init_h = result_df['simpleSolver'].values
    # result_df['bayesLinearSolver'] = run_one_method(d, 'bayesLinearSolver', breaks, init_h, dataloader)
    # result_df['bayesRBFSolver'] = run_one_method(d, 'bayesRBFSolver', breaks, init_h, dataloader)
    # result_df['logNormalSolver'] = run_one_method(d, 'logNormalSolver', breaks, init_h, dataloader)
    return result_df

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="This code will take in different possible values of parameters create design matrix dataframes that tell us the different parameter combinations that we will use to simulate the A matrix and the b vector and the solver settings.")
    parser.add_argument('--N', required=True, type=int, help='Number of rows in the A matrix that we will simulate')
    parser.add_argument('--G', required=True, type=float, help='Different possible values of the length of the gene')
    parser.add_argument('--time_traverse_gene', required=True, type=float, help='time (minutes) to traverse the gene')
    parser.add_argument('--label_time', required=True, type=int, help='length of labeling time (mins). this is the 3 * label_time per label in the kinetic barcoding experiment')
    parser.add_argument('--h_bin', required=True, type=float, help='length of each bin for which we will try to solve for the elongation rate')
    parser.add_argument('--seed', required=False, type=int, default=9999, help='Different possible values of the seed')
    parser.add_argument('--output_fn', required=True, type=str, help='Output of the analysis')
    parser.add_argument('--lambda_smooth', required=False, type=float, default=1, help='The lambda parameter for the smoothness constraint in the simpleSmoothSolver')
    args = parser.parse_args()
    helper.create_folder_for_file(args.output_fn)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    true_h = args.G / args.time_traverse_gene
    distance = true_h * args.label_time
    A, breaks = simulate_A(args.N, distance, args.G, args.h_bin)
    b = np.ones(args.N) * args.label_time
    print('Done with getting simulation data')
    result_df = pd.DataFrame()
    result_df['start'] = breaks[:-1]
    result_df['end'] = breaks[1:]
    result_df['true_h'] = true_h
    result_df.loc[result_df.shape[0]-1, 'end'] = args.G
    result_df = solve(A, b, breaks, result_df, init_h = true_h, lambda_smooth=args.lambda_smooth)  # for now, we train the model and gives it the true h as the initial value
    result_df.to_csv(args.output_fn, index=False, header=True, sep='\t', compression = 'gzip')
    print(f"Output saved to {args.output_fn}")


