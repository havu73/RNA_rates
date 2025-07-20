import numpy as np
import torch
SEED= 9999
np.random.seed(SEED)
import matplotlib.pyplot as plt
import pandas as pd
import sys
import os
import helper  # some functions to manage files and directories
import seaborn as sns
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) # this is supposed to let the code know that some of the import path is outside of this folder
from regression import fixedX_pwlr as fxpwlr  # this implementation is torch-based, which will be a lot faster than the numpy-based implementation
from .utils import drop_trailing_zeros, merge_intervals, filterOut_nan, findA
from solver_Ahb.train_loop import SimpleSolver, BayesianNormalSolver, BayesianRBFSolver, BayesianLogNormalSolver, SimpleSmoothSolver
from torch.utils.data import Dataset, DataLoader

ONE_KB = 1000

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
        _ = self.__remove_trailing_zeroes__() # remove trailing zeroes columns in A
        return

    def __len__(self):
        return len(self.data)

    def __remove_trailing_zeroes__(self) -> int:
        # Find the last column that isn't all zeros
        # all() along dim=0 checks each column
        last_nonzero = 0
        self.num_trailing_zeroes = 0
        for i in range(self.data.size(1) - 1, -1, -1):
            if not torch.all(self.data[:, i] == 0):
                last_nonzero = i
                break
            self.num_trailing_zeroes += 1
        # Return the tensor up to and including the last non-zero column
        self.clean_data = self.data[:, :last_nonzero + 1]
        return self.num_trailing_zeroes

    def __getitem__(self, idx):
        data_point = self.clean_data[idx]
        target = self.targets[idx]
        return data_point, target

    def avg_pred_h(self):
        '''
        Given A and b, calculate the average pred_h across all samples
        :return:
        '''
        dist_traveled = torch.sum(self.clean_data, dim = 1)
        pred_h = dist_traveled / self.targets
        return pred_h.mean(dim=0)

def run_one_method(d, method, breaks, init_h=1, dataloader=None, lambda_smooth = 1, num_trailing_zeroes = 0):
    total_d = d + num_trailing_zeroes
    nan_tensor = torch.tensor([np.nan]*total_d).cpu().numpy()
    trailing_nans = torch.tensor([torch.nan]*num_trailing_zeroes)
    if method == 'simpleSolver':
        Ssolver = SimpleSolver(d, init_h=init_h, dataloader=dataloader)  # solve for h such that A/h = b
        try:
            simple_h = Ssolver.solve()
            simple_h = torch.concatenate([simple_h, trailing_nans.to(simple_h.device)])
            return simple_h.cpu().numpy()
        except:
            return nan_tensor
    if method == 'simpleSmooth':
        SSS = SimpleSmoothSolver(d, init_h=init_h, dataloader=dataloader, lambda_smooth = lambda_smooth)  # solve for h such that A/h = b, with some smoothness constraint
        try:
            avg_h = dataloader.dataset.avg_pred_h()
            simple_smooth_h = SSS.solve(avg_h=avg_h)
            simple_smooth_h = torch.concatenate([simple_smooth_h, trailing_nans.to(simple_smooth_h.device)])
            return simple_smooth_h.cpu().numpy()
        except:
            return nan_tensor
    elif method == 'bayesLinearSolver':
        BNSolver = BayesianNormalSolver(d, init_h=init_h, dataloader=dataloader)  # solve for h such that A/h = b, with some prior distribution for h set by the solver
        try:
            bayes_linear_h = BNSolver.solve()
            bayes_linear_h = torch.concatenate([bayes_linear_h, trailing_nans.to(bayes_linear_h.device)])
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
            bayes_RBF_h = torch.concatenate([bayes_RBF_h, trailing_nans.to(bayes_RBF_h.device)])
            return bayes_RBF_h.cpu().numpy()
        except:
            return nan_tensor
    elif method == 'logNormalSolver':
        logNormalSolver = BayesianLogNormalSolver(d, init_h=init_h, dataloader=dataloader)
        try:
            logNormal_h = logNormalSolver.solve()
            logNormal_h = torch.concatenate([logNormal_h, trailing_nans])
            return logNormal_h.cpu().numpy()
        except:
            return nan_tensor
    else:
        return nan_tensor

def convert_df_to_KB(df, colnames = ['position']):
    for col in colnames:
        df[col] = df[col] / ONE_KB
    return df

def find_gap(X):
    '''
    Given a list of x-coordinates, find the gap between the x-coordinates (usually, it is either 1 or 0.001 depending on whether we are calculating the coverage in terms of bp or kb)
    But there can be cases when the gaps is variables (bc of subsampling)
    In that case we will return the median gap
    :param X:
    :return:
    '''
    X_diff = np.diff(X)
    if np.all(X_diff == X_diff[0]):
        return X_diff[0]
    else:
        return np.median(X_diff)

def calculate_culmulative_coverage(coverage_df):
    """
    Given the coverage_df, clean the coverage_df to make sure that the coverage at each timepoint is the culmulative coverage
    :param coverage_df: time-tagged coverage_df
    :param gapN: every gapN rows in the coverage_df, we will include a row to include into the piece-wise linear regression
    :return:
    """
    df_cumulative = coverage_df.drop('position', axis=1).copy()
    df_cumulative = df_cumulative.cumsum(axis=1)
    df_cumulative['position'] = coverage_df.loc[df_cumulative.index, 'position'] # put the position back in
    return df_cumulative

def find_valid_positions(df, time_idx, max_x0=None):
    '''
    Given coverage df (or, culmulative coverage_df), and a time_idx, find the positions that are valid (get rid of position that have trailing zeroes coverage)
    Also, valid means that it should be within the max_x0 (usually the end of the gene)
    :param coverage_df:
    :return: list of positions for which we have filtered out the trailing zeroes
    '''
    mask = (df[[time_idx]] !=0).any(axis=1)  # Create a mask where rows with any non-zero values are True
    results = df['position'].iloc[:int(mask[::-1].idxmax() + 1)]  # Reverse mask, find first True from end, slice DataFrame
    # also filter positions that are before max_x0
    if max_x0 is not None and np.isnan(max_x0) == False:
        results = results[results <= max_x0]
    return results


class estElong():
    def __init__(self, coverage_df, gtf_df, elongf_df=None, startT_idx=0, endT_idx=1, time=5, trim_endGene=False, subsample_frac=0.3, SEED=9999, convert_to_KB=True, regress_bin_bp=200, h_bin_bp=[1000], output_folder=None):
        '''
        :param coverage_df:
        :param gtf_df:
        :param startT_idx:
        :param endT_idx:
        :param time:
        :param max_segments:
        :param trim_endGene:
        :param subsample_frac:
        :param SEED:
        :param convert_to_KB:
        :param regress_bin_bp:
        '''
        self.coverage_df = coverage_df.copy()  # coverage df
        self.gtf_df = gtf_df  # gene feature df, gene feature always specify the feature coordinate and lengths in units of bp
        self.elongf_df = elongf_df
        if self.elongf_df is None:
            self.elongf_df = self.gtf_df.copy()
        try:
            self.elongf_df.rename(columns={'txrate': 'true_h'}, inplace=True)
        except:
            pass
        self.startT_idx = startT_idx
        self.endT_idx = endT_idx
        self.time = time
        self.trim_endGene = trim_endGene  # revise to see if it is necessary
        self.subsample_frac = subsample_frac  # revise to see if it is necessary
        self.SEED = SEED
        self.output_folder = output_folder
        helper.make_dir(self.output_folder)
        self.convert_to_KB = convert_to_KB
        self._converted_kb = False  # regardless of the user's choice, this is set to False at the object initiation
        self.regress_bin_bp = regress_bin_bp
        self.h_bin_bp = h_bin_bp # the binsize for the elongation rates calculation (numpy array bc we allow the functions to calculate with different binsizes)
        self.calculate_binsize()  # create self.regress_bin
        self.convert_to_KB_func()
        # this implines that the maximum elongation rates is 10 KB/min and the minimum elongation rate is 0.01 KB/min
        print ('The object assumes that the length and coordinates of features in both coverage_df and gtf_df are in the same units (bp or KB)')
        print ('If this assumption is NOT met, the object may NOT crash but may result in seriously wrong estimates of elongation rates')
        # here are the things that this object has to be able to do, eventually
        # 1. based on coverage (with splicing, read geenration, edge effects, etc.) recover the coverage without any of those
        # 2. Based on the full coverage, estimate the elongation rates based on the algorithm
        # 3. Estimate the limits of the elongation rates (in what case of the elongation rates that the method would fail to recover at the desired resolution)
        # 4. Draw the coverage and regression line estimate (for the purpose of visualization and sanity check) --> done!

    def calculate_binsize(self):
        '''
        This function does not conflict with the convert_to_kb function. I can call this function as many times as I need to.
        :return:
        '''
        self.regress_bin = self.regress_bin_bp
        self.h_bin = self.h_bin_bp
        if self.convert_to_KB:
            self.regress_bin = self.regress_bin_bp / ONE_KB
            self.h_bin = np.array(self.h_bin_bp) / ONE_KB
        return

    def convert_to_KB_func(self):
        '''
        Convert the coverage and gtf to KB if the user wants to
        :return:
        '''
        if self._converted_kb:  # this function should be returned once
            return
        if self.convert_to_KB:
            self.coverage_df = convert_df_to_KB(self.coverage_df, colnames=['position'])
            self.gtf_df = convert_df_to_KB(self.gtf_df, colnames=['start', 'end', 'length'])
            self.elongf_df = convert_df_to_KB(self.elongf_df, colnames=['start', 'end'])
            self.regress_bin = self.regress_bin_bp / ONE_KB
            self.h_bin = np.array(self.h_bin_bp) / ONE_KB  # numpy array bc we allow multiple options for the binsize
            try:  # if converted, and the full_coverage_df does not exist yet then we would not really need to call this function again bc full_coverage_df will copy the coverage_df['position']
                self.full_coverage_df = convert_df_to_KB(self.full_coverage_df, colnames=['position'])
            except:  # if the full_coverage_df does not exist yet!
                pass
        self._converted_kb = True
        return

    def find_even_breaks_along_gene(self, binsize):
        '''
        Given the binsize, find the breakpoints for the bins along the gene
        :param binsize:
        :return:
        '''
        self.convert_to_KB_func()  # convert all units of lengths to kb. This function is safe to call multiple times
        min_x0 = self.gtf_df['start'].min()
        max_x0 = self.gtf_df['end'].max()
        breaks = np.arange(min_x0, max_x0, binsize)
        breaks = np.append(breaks, np.inf)
        return breaks

    def find_h_breaks(self):
        self.h_breaks_list = []  # list of numpy arrays, each array contains the breakpoints for the bins used to calculate elongation rates
        for binsize in self.h_bin:
            h_breaks = self.find_even_breaks_along_gene(binsize)
            self.h_breaks_list.append(h_breaks)
        return


    def recover_full_coverage(self):
        '''
        Given that the coverage_df in the experiment may have gone through splicing, read generation, edge effects, etc., recover the coverage without any of those
        Note that self.coverge_df is time_tagged read coverage
        :return: self.full_coverage_df: should be culmulative coverage
        '''
        if hasattr(self, 'full_coverage_df'):
            return self.full_coverage_df
        self.convert_to_KB_func()  # convert all units of lengths to kb. This function is safe to call multiple times
        # for now, we only transform the coverage to culmulative coverage. Other transformations may be added later
        copy_coverage_df = self.coverage_df.drop('position', axis=1).copy()
        copy_coverage_df = drop_trailing_zeros(copy_coverage_df)  # find the rows where the coverage is zero for all time points, and remove them
        copy_coverage_df['position'] = self.coverage_df.loc[copy_coverage_df.index, 'position']  # put the position back in
        self.coverage_df = copy_coverage_df
        self.full_coverage_df = self.coverage_df.copy()
        self.full_coverage_df = calculate_culmulative_coverage(self.full_coverage_df)  # both full_coverage_df and coverage_df has the same nrows,and trailiing zeroes are removed
        GAP=5
        self.full_coverage_df = self.full_coverage_df[::GAP]
        self.coverage_df = self.coverage_df[::GAP]
        self.full_coverage_df.reset_index(drop=True, inplace=True)
        self.coverage_df.reset_index(drop=True, inplace=True)
        return self.full_coverage_df


    def _prepare_data_for_pwlr(self, time_idx):
        '''
        Look at the full_coverage_df and prepare the data: X: positions, Y: coverage at time_idx.
        Only select data from positions such that we get rid of trailing zeroes in coverage
        :param time_idx:
        :return:
        '''
        mask = (self.full_coverage_df[[time_idx]] != 0).any(axis=1)  # Create a mask where rows with any non-zero values are True
        end_idx = int(mask[::-1].idxmax() + 1)
        X = (self.full_coverage_df['position'].iloc[:end_idx].values).copy()  # Reverse mask, find first True from end, slice DataFrame
        Y = (self.full_coverage_df[time_idx].iloc[:end_idx].values).copy() # Reverse mask, find first True from end, slice DataFrame
        x_min = X.min()
        x_max = X.max()
        return X, Y, x_min, x_max

    def pwlr_from_full_coverage(self, recalculate=True, gapX_for_draw=50):
        '''
        Given the full coverage, do the piecewise linear regression
        :return:
        '''
        if (not recalculate) and hasattr(self, 'pwlr_dict'):
            return self.pwlr_dict
        self.pwlr_dict = {}  # keys: time points, values: (px, py) tuple. each px, py is an array of numbers corresponding to the endpoints of the piece-wise linear regression
        time_indices = [col for col in self.full_coverage_df.columns if col != 'position']
        for time_idx in time_indices:
            X, Y, x_min, x_max = self._prepare_data_for_pwlr(time_idx)  # parepare the data for the piecewise linear regression
            save_fn = f'{self.output_folder}/pwlr_{time_idx}.txt'
            if os.path.exists(save_fn):
                # pwlr_model = fxpwlr.PiecewiseLinearRegression(x_min=x_min, x_max=x_max, gap = self.regress_bin)
                # pwlr_model.load_model(save_fn)
                pwlr_model = fxpwlr.load_model(save_fn)
                self.pwlr_dict[time_idx] = pwlr_model
                # pwlr_model.draw_regression_results(X[::gapX_for_draw], save_fn=f'{self.output_folder}/pwlr_{time_idx}.png')  # draw a few points to verify that the regression calculation (x_to_y and y_to_x) is correct
                continue
            pwlr_model = fxpwlr.PiecewiseLinearRegression(x_min=x_min, x_max=x_max, gap = self.regress_bin)
            pwlr_model.fit(X, Y)
            self.pwlr_dict[time_idx] = pwlr_model
            pwlr_model.save_model(save_fn)
            # pwlr_model.draw_regression_results(X[::gapX_for_draw], save_fn=f'{self.output_folder}/pwlr_{time_idx}.png')  # draw a few points to verify that the regression calculation (x_to_y and y_to_x) is correct
        print('Done getting pwlr_dict')
        return self.pwlr_dict

    def find_x0_x1(self, recalculate=True):
        '''
        Given the full coverage, find the x0 and x1, which shows the start and end coordinates of a transcript within the m minutes of labeling
        :return:
        '''
        max_x0 = self.gtf_df[self.gtf_df['feature'] == 'RTR']['end'].max() # only consider positions that extend to the end of RTR
        if np.isnan(max_x0):
            max_x0 = None
        if (not recalculate) and hasattr(self, 'x0') and hasattr(self, 'x1'):
            return self.x0, self.x1
        # first, for each time point from startT_idx to endT_idx, do the piecewise linear regression
        self.pwlr_from_full_coverage(recalculate=recalculate) # self.pwlr_dict
        # second, find the x0 and x1 for each time point
        self.x0 = torch.tensor([])  # we will append the x0 and x1 for each time point
        self.x1 = torch.tensor([])
        for time_idx in range(self.startT_idx, self.endT_idx):
            nextT_idx = time_idx + 1
            x0 = find_valid_positions(self.full_coverage_df, time_idx, max_x0 = max_x0)  # we only takes positions for which there are no trailing zeroes in the coverage
            x0, y0 = self.pwlr_dict[time_idx].x_to_y(x0)  #all are 1D array, a
            # lready filted OUT the values of x0 that corresponds to a horizontal line
            x1, y1 = self.pwlr_dict[nextT_idx].y_to_x(y0)  # 1D array, same size as x0
            # now, x0_filtered and x1 should have the same length. But, we should filter out elements that are np.nan in x1 and the corresponding elements in x0_filtered
            x0_filtered, x1_filtered = filterOut_nan(x0, x1)
            self.x0 = torch.cat([self.x0, x0_filtered.clone().detach()])
            self.x1 = torch.cat([self.x1, x1_filtered.clone().detach()])
        self.save_x0_x1(f'{self.output_folder}/x0_x1.csv.gz')
        return

    def calculate_h_from_x0_x1(self, breaks=None, index=None):
        '''
        This function assumes that the followings are already calculated:
        1. self.x0 and self.x1: where the transcripts are supposed to be betwee two consecutive labelling time points
        2. self.regress_breaks: the endpoints of the features in the gene for which we will calculate the elongation rates
        :return: self.h: the elongation rates, with length equal to to len(self.regress_breaks)-1 --> elongation rates for each feature in the gene
        '''
        if breaks is None:
            breaks = [np.x0[0], np.inf]  # the first element is the start of the gene, the last element is np.inf
        n = len(self.x0)  # number of datapoints that we have to estimate elongation rate
        # n: number of samples (positions along the gene) and m: number of features/bins in the gene
        # based on x0,x1 and endpoints, find the coefficient matrix A (size n*m)
        A = findA(self.x0, self.x1, breaks)
        b = np.ones(n) * self.time
        avg_h = np.mean(np.sum(A, axis=1) / b)
        # 1, Get the dataloader
        dataset = CustomData(A, b)
        batch_size = 2048
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        d = A.shape[1] - dataset.num_trailing_zeroes
        result_df = pd.DataFrame()
        result_df['start'] = breaks[:-1]
        result_df['end'] = breaks[1:]
        result_df['simpleSmooth'] = run_one_method(d, 'simpleSmooth', breaks, init_h=avg_h, dataloader=dataloader, lambda_smooth=0.1, num_trailing_zeroes=dataset.num_trailing_zeroes)
        result_df['simpleSolver'] = run_one_method(d, 'simpleSolver', breaks, init_h=avg_h, dataloader=dataloader, num_trailing_zeroes=dataset.num_trailing_zeroes)
        result_df['bayesLinearSolver'] = run_one_method(d, 'bayesLinearSolver', breaks, init_h=avg_h, dataloader=dataloader, num_trailing_zeroes=dataset.num_trailing_zeroes)
        result_df['bayesRBFSolver'] = run_one_method(d, 'bayesRBFSolver', breaks, init_h=avg_h, dataloader=dataloader, num_trailing_zeroes=dataset.num_trailing_zeroes)
        result_df['logNormalSolver'] = run_one_method(d, 'logNormalSolver', breaks, init_h=avg_h, dataloader=dataloader, num_trailing_zeroes=dataset.num_trailing_zeroes)
        # if dataset.num_trailing_zeroes > 0: ## add that many rows of np.nan to the result_df

        result_df['index'] = index  # if we have multiple h_bin, the index can help differentiate the results
        return result_df

    def merge_predh_and_trueh(self, predh_df, predh_df_colnames=['simpleSmooth', 'simpleSolver', 'bayesLinearSolver', 'bayesRBFSolver', 'logNormalSolver', 'index']):
        if self.elongf_df is None:
            return predh_df
        predh_df = merge_intervals(self.elongf_df, predh_df, value1=['true_h'], value2=predh_df_colnames)
        return predh_df

    def estimate(self):
        '''
        Given the full coverage, estimate the elongation rates
        :return:
        '''
        print()
        print('Starting estimating the elongation rates')
        # first, transform the data from coverage_df into full_coverage_df such that it has the following effects:
        # 1. drop trailing zeroes
        # 2. culmulative coverage
        # 3. convert to KB if needed
        # 4. If the experiments are done such that there are read generation, splicing, etc., then the full_coverage_df should be the coverage without any of those  (Not implemented yet)
        self.recover_full_coverage()  # this function is designed to be executed only once
        print('Done recovering recover_full_coverage')
        # second, find the endpoints of bins for which we will (1) regress the coverage and (2) calculate the elongation rates
        self.find_h_breaks()  # self.h_breaks_list
        print('Done getting breaks')
        # third, calculate the x0 and x1, which shows the start and end coordinates of a transcript within the m minutes of labeling
        self.find_x0_x1()  # this will create self.x0 and self.x1
        print('Done find_x0_x1')
        self.draw_regression_lines(save_fn=f'{self.output_folder}/regression_lines.png', show=False)
        # fourth, calculate the elongation rates given the values of x0 and x1 between two time points
        hdf_list = []  # each element is a hdf for a different set of breaks
        for idx, breaks in enumerate(self.h_breaks_list):  # different ways that we create the breaks along which to calculate the elongation rates
            hdf = self.calculate_h_from_x0_x1(breaks, index=idx)  # hdf is a dataframe with columns ['start', 'end', 'elongation_rate'] --> elongation rates for each feature in the gene
            hdf = self.merge_predh_and_trueh(hdf)
            hdf_list.append(hdf)
        self.hdf = pd.concat(hdf_list, axis=0, ignore_index=True)  # start, end, init_h, pred_h
        print('Done calculating the elongation rates')
        print()
        return self.hdf

    def save_estimates(self, save_fn):
        '''
        Save the estimates of elongation rates. If it does not exist yet then calculate it based on the coverage data
        :param save_fn:
        :return:
        '''
        try:
            self.hdf
        except:  # self.hdf does not exist yet, calculate it
            self.estimate()
        helper.create_folder_for_file(save_fn)
        self.hdf.to_csv(save_fn, index=False, sep='\t', compression = 'gzip')
        return

    def save_x0_x1(self, save_fn):
        '''
        Save the x0 and x1 values. If it does not exist yet then calculate it based on the coverage data
        :param save_fn:
        :return:
        '''
        self.find_x0_x1(recalculate=False)  # this will create self.x0 and self.x1
        helper.create_folder_for_file(save_fn)
        df = pd.DataFrame({'x0': self.x0.detach().numpy(), 'x1': self.x1.detach().numpy()})
        df.to_csv(save_fn, header=True, index=False, sep='\t', compression='gzip')
        return

    def draw_regression_lines(self, save_fn=None, show=True):
        '''
        Draw the regression lines on top of the coverage data
        :param save_fn:
        :return:
        '''
        self.recover_full_coverage()  # create self.full_coverage_df from self.coverage_df
        self.pwlr_from_full_coverage(recalculate=False)  # for each time point from startT_idx to endT_idx, do the piecewise linear regression
        self.coverage_df = self.coverage_df.loc[self.full_coverage_df.index] # this line is needed to make sure that the coverage_df and full_coverage_df have the same index
        # dictionary with keys: time points, values: pwlr_model objects that has been trained on the data of culmulative coverage
        # for each pwlr_model, we can get access to the regression breakpoints through pwlr_model.px and pwlr_model.py
        plotGap_n = 10
        plot_df = self.coverage_df.loc[::plotGap_n]  # we do not need to plot every datapoints
        num_timepoints = len(self.pwlr_dict.keys())
        cmap = sns.color_palette('Blues', n_colors=num_timepoints)
        colors = [cmap[i] for i in range(num_timepoints)]
        fig, ax = plt.subplots(figsize=(10, 5))
        width = 0.85* find_gap(plot_df['position'])  # the width of the bars should never exceed the numerical distance between each of the x-axis value
        for time_idx in range(self.startT_idx, self.endT_idx+1):  # we want to plot the regression lines for the endT_idx as well
            bottom = np.zeros(len(plot_df['position']))
            if time_idx > 0:
                bottom = self.full_coverage_df.loc[::plotGap_n][time_idx-1].values
            ax.bar(plot_df['position'], plot_df[time_idx], color=colors[time_idx], bottom = bottom, alpha=0.8, label=time_idx, width=width, linewidth=0)
            pwlr_model = self.pwlr_dict[time_idx]
            px, py = pwlr_model.px.detach().numpy(), pwlr_model.py.detach().numpy()
            ax.plot(px, py, '-or')  # red
        ax.set_xlim(0, self.full_coverage_df['position'].max())
        ax.set_ylim(0, self.full_coverage_df.max().max())
        ax.legend()
        if save_fn is not None:
            helper.create_folder_for_file(save_fn)
            plt.savefig(save_fn)
        if show:
            plt.show()
        return

    def draw_distance_travelled(self, save_fn=None, show=True):
        '''
        Draw the distance travelled by the RNA molecules at each time point
        :param save_fn:
        :return:
        '''
        self.find_x0_x1(recalculate=False)  # this will create self.x0 and self.x1
        distances = self.x1 - self.x0
        # two subplots, one for the distance travelled at each time point, and the other for the histogram of the distances
        fig, axes = plt.subplots(2, 1, figsize=(10, 10))
        axes[0].plot(self.x0, distances, 'o')
        axes[0].set_xlabel('Sample index')
        axes[0].set_ylabel('Distance travelled (KB)')
        axes[1].hist(distances, bins=50)
        axes[1].set_xlabel('Distance travelled (KB)')
        axes[1].set_ylabel('Frequency')
        if save_fn is not None:
            helper.create_folder_for_file(save_fn)
            plt.savefig(save_fn)
        if show:
            plt.show()
        return





