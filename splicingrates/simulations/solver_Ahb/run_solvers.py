import pandas as pd
import numpy as np
import torch
from .train_loop import SimpleSolver, BayesianNormalSolver, BayesianRBFSolver, BayesianLogNormalSolver, SimpleSmoothSolver
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


def run_one_method(d, method, breaks, init_h, dataloader):
    nan_tensor = torch.tensor([np.nan]*d).cpu().numpy()
    if method == 'simpleSolver':
        Ssolver = SimpleSolver(d, init_h=init_h, dataloader=dataloader)  # solve for h such that A/h = b
        try:
            simple_h = Ssolver.solve()
            return simple_h.cpu().numpy()
        except:
            return nan_tensor
    elif method == 'bayesLinearSolver':
        BNSolver = BayesianNormalSolver(d, init_h=init_h, dataloader=dataloader)  # solve for h such that A/h = b, with some prior distribution for h set by the solver
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
        B_RBFSolver = BayesianRBFSolver(d, init_h=init_h, dataloader=dataloader, coords=breaks[:-1] + h_bins * 0.5)
        try:
            bayes_RBF_h = B_RBFSolver.solve()
            return bayes_RBF_h.cpu().numpy()
        except:
            return nan_tensor
    elif method == 'logNormalSolver':
        logNormalSolver = BayesianLogNormalSolver(d, init_h=init_h, dataloader=dataloader)
        try:
            logNormal_h = logNormalSolver.solve()
            return logNormal_h.cpu().numpy()
        except:
            return nan_tensor
    else:
        return nan_tensor


def solve(A, b, breaks, init_h=1, use_simpleSolver_init=False):
    '''
    Solve for the elongation rates given A and b
    :param A: (n, m) where n is the number of samples and m is the number of features
    :param b: (n,) where n is the number of samples
    :param breaks: (m+1,) where m is the number of features
    :return: simpleSolver, bayesianLinearSolver, bayesianRBFsolver
    '''
    print('Solving for the elongation rates')
    result_df = pd.DataFrame()
    result_df['start'] = breaks[:-1]
    result_df['end'] = breaks[1:]
    # 1, Get the dataloader
    dataset = CustomData(A, b)
    batch_size = 2048
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    d = A.shape[1]
    # 2, Solve and save
    result_df['simpleSolver'] = run_one_method(d, 'simpleSolver', breaks, init_h, dataloader)
    if use_simpleSolver_init:  # if we want to use the output of simpleSolver as the initial values for the other solvers
        # if there are not negative entries in the result of simpleSolver, we can use it as the initial value for the other solvers
        if np.all(result_df['simpleSolver'] >= 0):
            init_h = result_df['simpleSolver'].values
    result_df['bayesLinearSolver'] = run_one_method(d, 'bayesLinearSolver', breaks, init_h, dataloader)
    result_df['bayesRBFSolver'] = run_one_method(d, 'bayesRBFSolver', breaks, init_h, dataloader)
    result_df['logNormalSolver'] = run_one_method(d, 'logNormalSolver', breaks, init_h, dataloader)
    return result_df
