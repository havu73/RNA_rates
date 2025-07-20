import pandas as pd
import numpy as np
import argparse
import helper
import os
import itertools
import sys

from torch.optim.optimizer import required

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import transcription.simulate_multiple_experiments as sim
def generate_combinations(lists):
    # Generate all combinations using itertools.product
    combinations = list(itertools.product(*lists))
    return combinations


def generate_combinations_dict(lists_dict):
	# Extract keys and values from the dictionary
	keys = lists_dict.keys()
	values = lists_dict.values()
	# Generate all combinations using itertools.product
	combinations = list(itertools.product(*values))
	# Convert each combination into a dictionary
	combinations_dicts = [dict(zip(keys, combination)) for combination in combinations]
	return combinations_dicts


def get_design_df_constant(N, G, time_traverse_gene, label_time, h_bin, seed, output_folder, save_fn):
	'''
	:param N: list of number of rows in the A matrix that we will simulate
	:param G: list of different possible values of the length of the gene
	:param time_traverse_gene: list of time (minutes) to traverse the gene
	:param label_time: list of length of labeling time (mins). this is the 3 * label_time per label in the kinetic barcoding experiment
	:param h_bin: list of length of each bin for which we will try to solve for the elongation rate
	:param seed: list of different possible values of the seed
	:param output_folder: The folder where we will save the results of all the runs.
	:param save_fn: The file where we will save the design dataframe.
	:return: a dataframe that contains all the possible combinations of the parameters
	'''
	design_df = pd.DataFrame(columns=['N', 'G', 'time_traverse_gene', 'label_time', 'h_bin', 'seed'])
	entry_dict = {'N': N, 'G': G, 'time_traverse_gene': time_traverse_gene, 'label_time': label_time, 'h_bin': h_bin, 'seed': seed}
	rows = generate_combinations_dict(entry_dict)  # list of dictionaries, each dictionary can be used as row in design_fn
	for row_idx, row in enumerate(rows):
		design_df.loc[row_idx] = row
	design_df['output_fn'] = design_df.apply(lambda x: os.path.join(output_folder, f'result_{x.name}.txt.gz'), axis=1)  # apply to each row
	design_df.reset_index(drop=True, inplace=True)
	design_df.to_csv(save_fn, header =True, index = False, sep = '\t')
	return design_df

def get_design_df_constant(N, G, time_traverse_gene, label_time, h_bin, seed, output_folder, save_fn, lambda_smooth):
	'''
	:param N: list of number of rows in the A matrix that we will simulate
	:param G: list of different possible values of the length of the gene
	:param time_traverse_gene: list of time (minutes) to traverse the gene
	:param label_time: list of length of labeling time (mins). this is the 3 * label_time per label in the kinetic barcoding experiment
	:param h_bin: list of length of each bin for which we will try to solve for the elongation rate
	:param seed: list of different possible values of the seed
	:param output_folder: The folder where we will save the results of all the runs.
	:param save_fn: The file where we will save the design dataframe.
	:return: a dataframe that contains all the possible combinations of the parameters
	'''
	design_df = pd.DataFrame(columns=['N', 'G', 'time_traverse_gene', 'label_time', 'h_bin', 'seed', 'lambda_smooth'])
	entry_dict = {'N': N, 'G': G, 'time_traverse_gene': time_traverse_gene, 'label_time': label_time, 'h_bin': h_bin, 'seed': seed, 'lambda_smooth': lambda_smooth}
	rows = generate_combinations_dict(entry_dict)  # list of dictionaries, each dictionary can be used as row in design_fn
	for row_idx, row in enumerate(rows):
		design_df.loc[row_idx] = row
	design_df['output_fn'] = design_df.apply(lambda x: os.path.join(output_folder, f'result_{x.name}.txt.gz'), axis=1)  # apply to each row
	design_df.reset_index(drop=True, inplace=True)
	design_df.to_csv(save_fn, header =True, index = False, sep = '\t')
	return design_df

def simulate_h_from_gamma(G, time_traverse_gene, vary_gamma_scale, h_bin, seed=9999):
	'''
	Given parameters about the gene and how we want to vary the ground truth elongation rates, we will create an array of elongation rates fold for each bin
	:param G: gene length in KB
	:param time_traverse_gene: time to traverse the gene in minutes --> used to calculate the bg mean elongation rate
	:param vary_gamma_scale: scale of the gamma distribution for the elongation rate var = mean*scale
	:param h_bin:
	:param seed:
	:return:
	'''
	bg_h = G / time_traverse_gene
	np.random.seed(seed)
	num_bins = int(G / h_bin)
	h = np.random.gamma(shape=bg_h/vary_gamma_scale, scale=vary_gamma_scale, size=num_bins)
	return h

def get_fold_list_for_elongf_df(G=5, vary_bin_kb=1, time_traverse_gene=5, vary_fold=1, vary_middle=False, vary_across=False, vary_gamma_scale=1):
	'''
	Given the length of the gene and the length of the middle segment, return the elongation fold list for 3 segments
	:param G:
	:param vary_bin_kb:
	:return:
	'''
	if vary_middle:
		first_kb = (G - vary_bin_kb) / 2
		length_fold_list = [first_kb, vary_bin_kb, first_kb]
		bg_h = G / time_traverse_gene
		elong_fold_list = [bg_h, bg_h * vary_fold, bg_h]
		return length_fold_list, elong_fold_list
	elif vary_across: # vary elongation rate across genes, based on a gamma distribution
		num_bins = int(G / vary_bin_kb)
		length_fold_list = [vary_bin_kb] * num_bins
		elong_fold_list = simulate_h_from_gamma(G, time_traverse_gene, vary_gamma_scale, vary_bin_kb)
		return length_fold_list, elong_fold_list
	else: # uniform elongation rate across genes
		length_fold_list = [G]
		bg_h = G / time_traverse_gene
		elong_fold_list = [bg_h]
		return length_fold_list, elong_fold_list



def custom_elongf_df_varyMiddle(G, time_traverse_gene, vary_bin_kb=1, fold_change=2, elongf_fn=None):
	'''
	:param G: G, in KB
	:param time_traverse_gene: time_traverse_gene (this is used to calculate the background elongation rates)
	:param vary_bin_kb: size of the middle part of the gene that will get a different elongation rate
	:param fold_change: fold change in the elongation rate in the middle of the gene
	:param output_folder: output_folder where we will save the elongf_df
	:param mean_elong_rate: default_elong_rate
	:param var_elong_rate: var_elong_rate
	:param output_fn: output_fn
	:param length_fold_list: length_fold_list
	:param elong_fold_list: elong_fold_list
	:return: a dataframe that contains all the possible combinations of the parameters
	'''
	ONE_KB = 1000
	# For a gene of length G, get the length of the first, middle and last segment, given that the middle segment is vary_bin_kb long
	length_fold_list, elong_fold_list = get_fold_list_for_elongf_df(G=G, vary_bin_kb=vary_bin_kb, time_traverse_gene=time_traverse_gene, vary_fold=fold_change, vary_middle=True, vary_across=False)
	elongf_df = sim.create_elongf_df(elong_fold_list=elong_fold_list, length_fold_list=length_fold_list, SIM_FEAT_LEN=ONE_KB)  # the last part extends to the end of the gene and will have the same elongation rate as the one before it
	if not elongf_fn is None:
		helper.create_folder_for_file(elongf_fn)
		elongf_df.to_csv(elongf_fn, header =True, index = False, sep = '\t')
	return elongf_df

def custom_elongf_df_varyAcross(G, time_traverse_gene, vary_bin_kb=1, vary_gamma_scale=1, elongf_fn=None):
	'''
	:param G: G, in KB
	:param time_traverse_gene: time_traverse_gene (this is used to calculate the background elongation rates)
	:param vary_bin_kb: size of the middle part of the gene that will get a different elongation rate
	:param elongf_fn: elongf_fn
	:param vary_gamma_scale: scale parameter for gamma distribution. mean = scale * shape, var = scale^2 * shape
	:return: a dataframe that contains all the possible combinations of the parameters
	'''
	ONE_KB = 1000
	# For a gene of length G, get the length of the first, middle and last segment, given that the middle segment is vary_bin_kb long
	length_fold_list, elong_fold_list = get_fold_list_for_elongf_df(G=G, vary_bin_kb=vary_bin_kb, time_traverse_gene=time_traverse_gene, vary_middle=False, vary_across=True, vary_gamma_scale=vary_gamma_scale)
	elongf_df = sim.create_elongf_df(elong_fold_list=elong_fold_list, length_fold_list=length_fold_list, SIM_FEAT_LEN=ONE_KB)  # the last part extends to the end of the gene and will have the same elongation rate as the one before it
	if not elongf_fn is None:
		helper.create_folder_for_file(elongf_fn)
		elongf_df.to_csv(elongf_fn, header =True, index = False, sep = '\t')
	return elongf_df


def get_design_df_vary_middle(N, G, time_traverse_gene, label_time, vary_bin_kb, vary_fold, h_bin, output_folder, lambda_smooth):
	'''
	:param N: list of number of rows in the A matrix that we will simulate
	:param G: list of different possible values of the length of the gene
	:param time_traverse_gene: list of time (minutes) to traverse the gene
	:param label_time: list of length of labeling time (mins). this is the 3 * label_time per label in the kinetic barcoding experiment
	:param vary_bin_kb: list of length of the segment in the middle of the gene that we will make the elongation rate different from the rest of the gene
	:param vary_fold: list of the fold change in the elongation rate in the middle of the gene
	:param h_bin: list of length of each bin for which we will try to solve for the elongation rate
	:param output_folder: The folder where we will save the results of all the runs.
	:return: a dataframe that contains all the possible combinations of the parameters
	'''
	design_df = pd.DataFrame(columns=['N', 'G', 'time_traverse_gene', 'label_time', 'vary_bin_kb', 'vary_fold', 'h_bin', 'lambda_smooth'])
	entry_dict = {'N': N, 'G': G, 'time_traverse_gene': time_traverse_gene, 'label_time': label_time, 'vary_bin_kb': vary_bin_kb, 'vary_fold': vary_fold, 'h_bin': h_bin, 'lambda_smooth': lambda_smooth}
	rows = generate_combinations_dict(entry_dict)  # list of dictionaries, each dictionary can be used as row in design_fn
	for row_idx, row in enumerate(rows):
		design_df.loc[row_idx] = row
		elongf_fn = os.path.join(output_folder, f'elongf_{row_idx}.txt.gz')
		design_df.loc[row_idx, 'elongf_fn'] = elongf_fn
		custom_elongf_df_varyMiddle(row['G'], row['time_traverse_gene'], row['vary_bin_kb'], row['vary_fold'], elongf_fn)
	design_df['output_fn'] = design_df.apply(lambda x: os.path.join(output_folder, f'result_{x.name}.txt.gz'), axis=1)  # apply to each row
	design_df.reset_index(drop=True, inplace=True)
	return design_df

def get_design_df_vary_across(N, G, time_traverse_gene, label_time, vary_gamma_scale, vary_bin_kb, h_bin, output_folder, lambda_smooth):
	'''
	:param N: list of number of rows in the A matrix that we will simulate
	:param G: list of different possible values of the length of the gene
	:param time_traverse_gene: list of time (minutes) to traverse the gene
	:param label_time: list of length of labeling time (mins). this is the 3 * label_time per label in the kinetic barcoding experiment
	:param vary_gamma_scale: list of the scale of the gamma distribution for the elongation rate
	:param vary_bin_kb: list of length of the segment in the middle of the gene that we will make the elongation rate different from the rest of the gene
	:param h_bin: list of length of each bin for which we will try to solve for the elongation rate
	:param output_folder: The folder where we will save the results of all the runs.
	:return: a dataframe that contains all the possible combinations of the parameters
	'''
	design_df = pd.DataFrame(columns=['N', 'G', 'time_traverse_gene', 'label_time', 'vary_gamma_scale', 'vary_bin_kb', 'h_bin', 'lambda_smooth'])
	entry_dict = {'N': N, 'G': G, 'time_traverse_gene': time_traverse_gene, 'label_time': label_time, 'vary_gamma_scale': vary_gamma_scale, 'vary_bin_kb': vary_bin_kb, 'h_bin': h_bin, 'lambda_smooth': lambda_smooth}
	rows = generate_combinations_dict(entry_dict)  # list of dictionaries, each dictionary can be used as row in design_fn
	for row_idx, row in enumerate(rows):
		design_df.loc[row_idx] = row
		elongf_fn = os.path.join(output_folder, f'elongf_{row_idx}.txt.gz')
		design_df.loc[row_idx, 'elongf_fn'] = elongf_fn
		custom_elongf_df_varyAcross(row['G'], row['time_traverse_gene'], row['vary_bin_kb'], row['vary_gamma_scale'], elongf_fn)
	design_df['output_fn'] = design_df.apply(lambda x: os.path.join(output_folder, f'result_{x.name}.txt.gz'), axis=1)  # apply to each row
	design_df.reset_index(drop=True, inplace=True)
	return design_df

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description = "This code will take in different possible values of parameters create design matrix dataframes that tell us the different parameter combinations that we will use to simulate the A matrix and the b vector and the solver settings.")
	parser.add_argument('--N', required = False, type = int, nargs='+', default=[1000, 10000, 100000],  help='Number of rows in the A matrix that we will simulate')
	parser.add_argument('--G', required = False, type = float, nargs='+', default=[3, 15, 30, 75], help='Different possible values of the length of the gene')
	parser.add_argument('--time_traverse_gene', required = False, type = float, nargs='+', default=[3, 6, 15, 30, 60], help='time (minutes) to traverse the gene')
	parser.add_argument('--label_time', required = False, type = int, nargs='+', default=[3, 6, 15, 30], help='length of labeling time (mins). this is the 3 * label_time per label in the kinetic barcoding experiment')
	parser.add_argument('--vary_bin_kb', required=False, type=float, nargs='+',default=[0.002, 0.1, 0.2, 0.5, 1, 2, 3], help='length of the segment in the middle of the gene that we will make the elongation rate different from the rest of the gene')
	parser.add_argument('--vary_fold', required=False, type=float, nargs='+', default=[0.25, 0.5, 2, 4, 8], help='The fold change in the elongation rate in the middle of the gene')
	parser.add_argument('--vary_gamma_scale', required=False, type=float, nargs='+', default=[0.5, 1, 2], help='The scale of the gamma distribution for the elongation rate')
	parser.add_argument('--h_bin', required = False, type = float, nargs='+', default=[0.001, 0.01, 0.1, 1, 2, 5], help='length of each bin for which we will try to solve for the elongation rate')
	parser.add_argument('--vary_middle', action='store_true', help='if this flag is present, we will vary the elongation rate in the middle of the gene')
	parser.add_argument('--vary_across', action='store_true', help='if this flag is present, we will vary the elongation rate across the gene')
	parser.add_argument('--lambda_smooth', required=False, type=float, nargs='+', default=[1], help='The lambda for the smoothness constraint (in simpleSmoothSolver)')
	parser.add_argument('--seed', required = False, type = int, nargs='+', default=[9999], help='Different possible values of the seed')
	parser.add_argument('--save_fn', required=True, type=str, help='The file where we will save the design dataframe.')
	parser.add_argument('--output_folder', required=True, type=str, help='The folder where we will save the results of all the runs.')
	args = parser.parse_args()
	helper.create_folder_for_file(args.save_fn)
	print ('create_design_parameters.py: Done getting command line argument')
	if args.vary_middle:
		design_df = get_design_df_vary_middle(args.N, args.G, args.time_traverse_gene, args.label_time, args.vary_bin_kb, args.vary_fold, args.h_bin, args.output_folder, args.lambda_smooth)
	elif args.vary_across:
		design_df = get_design_df_vary_across(args.N, args.G, args.time_traverse_gene, args.label_time, args.vary_gamma_scale, args.vary_bin_kb, args.h_bin, args.output_folder, args.lambda_smooth)
	else:
		design_df = get_design_df_constant(args.N, args.G, args.time_traverse_gene, args.label_time, args.h_bin, args.seed,args.output_folder, args.save_fn, args.lambda_smooth)
	design_df.to_csv(args.save_fn, header =True, index = False, sep = '\t')
	print('Done!')

# python create_design_matrix.py --N 15000 --G 15 --vary_middle --save_fn '/gladstone/engelhardt/lab/hvu//RNA_rates/solver_analysis/vary_middle/design_matrix.csv' --output_folder '/gladstone/engelhardt/lab/hvu//RNA_rates/solver_analysis/vary_middle/'
# python create_design_matrix.py --N 15000 --G 15 --vary_middle --vary_fold 0.001 --label_time 5 --save_fn '/gladstone/engelhardt/lab/hvu//RNA_rates/solver_analysis/pause_middle/design_matrix.csv' --output_folder '/gladstone/engelhardt/lab/hvu//RNA_rates/solver_analysis/pause_middle/'
# python create_design_matrix.py  --N 15000 --G 15 --vary_across --vary_gamma_scale 0.5 1 2 --label_time 5 --save_fn '/gladstone/engelhardt/lab/hvu//RNA_rates/solver_analysis/vary_across/design_matrix.csv' --output_folder '/gladstone/engelhardt/lab/hvu//RNA_rates/solver_analysis/vary_across/'