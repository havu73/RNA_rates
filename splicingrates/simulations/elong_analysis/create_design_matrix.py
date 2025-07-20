import pandas as pd
import numpy as np
import argparse

from docutils.nodes import description

import helper
import os
import itertools
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) # this is supposed to let the code know that some of the import path is outside of this folder
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

def get_constant_elong_design_df(lambda_init, burst_size, insertsize_min, insertsize_max, read_length, seed, num_exons, feat_length, time_traverse_gene, output_folder, save_fn, time, h_bounds=(0.1, 10)):
	'''
	:param lambda_init: list of average number of init events per minute
	:param burst_size: list of average number of new transcripts in each burst
	:param insertsize_min: list of minimum read length filter
	:param insertsize_max: list of maximum read length filter
	:param read_length: list of read length filter
	:param seed: list of seed
	:param num_exons: number of exon in the gene
	:param feat_length: list of possible length of a feature in the gene (we probably have 3 features in the gene)
	:return: a dataframe that contains all the possible combinations of the parameters
	'''
	design_df = pd.DataFrame(columns=['lambda_init', 'burst_size', 'insertsize_min', 'insertsize_max', 'read_length', 'seed', 'constant_elong_rate', 'length_fold_list', 'time_traverse_gene'])
	num_features = int(num_exons*2-1)
	length_comb_list = [tuple([x]*num_features) for x in feat_length]
	entry_dict = {'lambda_init': lambda_init, 'burst_size': burst_size, 'insertsize_min': insertsize_min, 'insertsize_max': insertsize_max, 'read_length': read_length, 'seed': seed, 'length_fold_list': length_comb_list, 'time': time, 'time_traverse_gene': time_traverse_gene}
	rows = generate_combinations_dict(entry_dict)  # list of dictionaries, each dictionary can be used as row in design_fn
	for row_idx, row in enumerate(rows):
		design_df.loc[row_idx] = row
	design_df['constant_elong_rate'] = design_df.apply(lambda x: np.sum(np.array(x['length_fold_list'])/ x['time_traverse_gene']), axis = 1)
	design_df = design_df[(design_df['constant_elong_rate']>=h_bounds[0]) & (design_df['constant_elong_rate'] <=h_bounds[1])]
	design_df.reset_index(drop=True, inplace=True)
	design_df['output_folder'] = design_df.apply(lambda x: os.path.join(output_folder, f'result_{x.name}'), axis=1)  # apply to each row
	design_df.to_csv(save_fn, header =True, index = False, sep = '\t')
	return design_df

def create_custom_elongf_df(gene_length, time_traverse_gene, vary_middle= False, vary_across=False, bin_size_bp=1000, vary_mid_fold=2, vary_sd= 0, output_folder='./'):
	'''
	:param gene_length: gene_length, in KB
	:param mean_elong_rate: default_elong_rate
	:param var_elong_rate: var_elong_rate
	:param output_fn: output_fn
	:param length_fold_list: length_fold_list
	:param elong_fold_list: elong_fold_list
	:return: a dataframe that contains all the possible combinations of the parameters
	'''
	result_df = pd.DataFrame(columns=['start', 'end', 'txrate', 'length'])
	ONE_KB = 1000
	bin_size_kb = bin_size_bp/ONE_KB
	avg_elong_rate = gene_length/time_traverse_gene
	bin_list = [] ## length of bins
	elong_list = [] ## elongation rate of each bin
	if vary_middle:
		first_bin = float(gene_length/2 - bin_size_kb/2)
		bin_list = [first_bin, bin_size_kb, gene_length]
		elong_list = [avg_elong_rate, avg_elong_rate*vary_mid_fold, avg_elong_rate]
	elif vary_across:
		num_bins = int(gene_length/bin_size_kb)
		bin_list = [bin_size_kb] * num_bins
		elong_list = np.random.normal(avg_elong_rate, vary_sd, num_bins)
	# the last bin should end at np.inf and has the elongation rate of the last bin
	bin_list[-1] = np.inf
	result_df['length'] = bin_list
	result_df['txrate'] = elong_list
	result_df['start'] = result_df['length'].cumsum().shift(1).fillna(0).astype(int)
	result_df['end'] = result_df['start']+result_df['length']
	helper.make_dir(output_folder)
	result_df.to_csv(os.path.join(output_folder, 'input_elongf_df.csv'), header =True, index = False, sep = '\t')
	return result_df



def create_dummies_files(output_folder):
	'''
	This function is used to create dummy files so that snakemake would not run the job
	:param output_folder:
	:return:
	'''
	df = pd.DataFrame()
	df.to_csv(os.path.join(output_folder, 'coverage_df.csv.gz'), header =True, index = False, sep = '\t', compression='gzip')
	df.to_csv(os.path.join(output_folder, 'pred_h.csv.gz'), header =True, index = False, sep = '\t', compression='gzip')
	df.to_csv(os.path.join(output_folder, 'x0_x1.csv.gz'), header =True, index = False, sep = '\t', compression='gzip')
	return

def get_vary_Middle_elong_design_df(lambda_init, burst_size, insertsize_min, insertsize_max, read_length, seed, num_exons, feat_length, time_traverse_gene, output_folder, save_fn, time, h_bounds=(0.1, 100), vary_bin_bp=1000, vary_fold=2, pair_end=False):
	'''
	:param lambda_init: list of average number of init events per minute
	:param burst_size: list of average number of new transcripts in each burst
	:param insertsize_min: list of minimum read length filter
	:param insertsize_max: list of maximum read length filter
	:param read_length: list of read length filter
	:param seed: list of seed
	:param num_exons: number of exon in the gene
	:param feat_length: list of possible length of a feature in the gene (we probably have 3 features in the gene)
	:return: a dataframe that contains all the possible combinations of the parameters
	'''
	design_df = pd.DataFrame(columns=['lambda_init', 'burst_size', 'insertsize_min', 'insertsize_max', 'read_length', 'seed', 'constant_elong_rate', 'length_fold_list', 'time_traverse_gene', 'pair_end'])
	num_features = int(num_exons*2-1)
	length_comb_list = [tuple([x]*num_features) for x in feat_length]
	entry_dict = {'lambda_init': lambda_init, 'burst_size': burst_size, 'insertsize_min': insertsize_min, 'insertsize_max': insertsize_max, 'read_length': read_length, 'seed': seed, 'length_fold_list': length_comb_list, 'time': time, 'time_traverse_gene': time_traverse_gene, 'pair_end': [pair_end]}
	rows = generate_combinations_dict(entry_dict)  # list of dictionaries, each dictionary can be used as row in design_fn
	for row_idx, row in enumerate(rows):
		design_df.loc[row_idx] = row
	design_df['output_folder'] = design_df.apply(lambda x: os.path.join(output_folder, f'result_{x.name}'), axis=1)  # apply to each row
	for idx, row in design_df.iterrows():
		elongf_df = create_custom_elongf_df(gene_length=np.sum(row['length_fold_list']), time_traverse_gene=row['time_traverse_gene'], bin_size_bp=vary_bin_bp, fold_change=vary_fold, vary_middle=True, output_folder=row['output_folder'])
		if elongf_df['txrate'].min() < h_bounds[0] or elongf_df['txrate'].max() > h_bounds[1]:
			design_df.loc[idx, 'valid_elong_rate'] = False
			create_dummies_files(row['output_folder'])
		design_df.loc[idx, 'elongf_fn'] = os.path.join(row['output_folder'], 'input_elongf_df.csv')
	design_df['avg_elong_rate'] = design_df.apply(lambda x: np.sum(np.array(x['length_fold_list'])/ x['time_traverse_gene']), axis = 1)
	design_df['elongf_fn'] = design_df['output_folder'] + '/input_elongf_df.csv'  # apply to each row
	filterOUT_df = design_df[(design_df['avg_elong_rate']<h_bounds[0]) | (design_df['avg_elong_rate'] >h_bounds[1])]
	if filterOUT_df.shape[0]>0:
		for idx, row in filterOUT_df.iterrows():
			create_dummies_files(row['output_folder'])
	design_df.reset_index(drop=True, inplace=True)
	design_df.to_csv(save_fn, header =True, index = False, sep = '\t')
	return design_df

def get_vary_Across_elong_design_df(lambda_init, burst_size, insertsize_min, insertsize_max, read_length, seed, num_exons, feat_length, time_traverse_gene, output_folder, save_fn, time, h_bounds=(0.1, 100), vary_bin_bp=1000, vari_sd = 0, pair_end=False):
	'''
	:param lambda_init: list of average number of init events per minute
	:param burst_size: list of average number of new transcripts in each burst
	:param insertsize_min: list of minimum read length filter
	:param insertsize_max: list of maximum read length filter
	:param read_length: list of read length filter
	:param seed: list of seed
	:param num_exons: number of exon in the gene
	:param feat_length: list of possible length of a feature in the gene (we probably have 3 features in the gene)
	:return: a dataframe that contains all the possible combinations of the parameters
	'''
	design_df = pd.DataFrame(columns=['lambda_init', 'burst_size', 'insertsize_min', 'insertsize_max', 'read_length', 'seed', 'constant_elong_rate', 'length_fold_list', 'time_traverse_gene', 'pair_end'])
	num_features = int(num_exons*2-1)
	length_comb_list = [tuple([x]*num_features) for x in feat_length]
	entry_dict = {'lambda_init': lambda_init, 'burst_size': burst_size, 'insertsize_min': insertsize_min, 'insertsize_max': insertsize_max, 'read_length': read_length, 'seed': seed, 'length_fold_list': length_comb_list, 'time': time, 'time_traverse_gene': time_traverse_gene, 'pair_end': [pair_end]}
	rows = generate_combinations_dict(entry_dict)  # list of dictionaries, each dictionary can be used as row in design_fn
	for row_idx, row in enumerate(rows):
		design_df.loc[row_idx] = row
	design_df['output_folder'] = design_df.apply(lambda x: os.path.join(output_folder, f'result_{x.name}'), axis=1)  # apply to each row
	design_df['valid_elong_rate'] = True
	for idx, row in design_df.iterrows():
		elongf_df = create_custom_elongf_df(gene_length=np.sum(row['length_fold_list']), time_traverse_gene=row['time_traverse_gene'], bin_size_bp=vary_bin_bp, vary_sd=vari_sd, vary_across=True, output_folder=row['output_folder'])
		if elongf_df['txrate'].min() < h_bounds[0] or elongf_df['txrate'].max() > h_bounds[1]:
			design_df.loc[idx, 'valid_elong_rate'] = False
			create_dummies_files(row['output_folder'])
		design_df.loc[idx, 'elongf_fn'] = os.path.join(row['output_folder'], 'input_elongf_df.csv')
	design_df['avg_elong_rate'] = design_df.apply(lambda x: np.sum(np.array(x['length_fold_list'])/ x['time_traverse_gene']), axis = 1)
	design_df['elongf_fn'] = design_df['output_folder'] + '/input_elongf_df.csv'  # apply to each row
	design_df.reset_index(drop=True, inplace=True)
	design_df['burst_size'] = design_df['burst_size'].astype(int)
	design_df['G'] = design_df['length_fold_list'].apply(sum)
	design_df.to_csv(save_fn, header =True, index = False, sep = '\t')
	return design_df


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description = "This code will take create dataframes that tell us the different parameter combinations that we will use to simulate the transcription process, and where we will find the results file for each of the parameter combinations.")
	parser.add_argument('--PDB', action='store_true' , help='Whether we simulate the transcription with PDB or not')
	parser.add_argument('--lambda_init', required = False, type = float, nargs='+', default=[2, 10, 30, 60, 120],  help='Average number of init events per minute')
	parser.add_argument('--burst_size', required = False, type = float, nargs='+', default=[15], help='Average number of new transcripts in each burst')
	parser.add_argument('--insertsize_min', required = False, type = int, nargs='+', default=[200], help='minimum read length filter')
	parser.add_argument('--insertsize_max', required = False, type = int, nargs='+', default=[300], help='maximum read length filter')
	parser.add_argument('--read_length', required = False, type = int, nargs='+', default=[100], help='read length flter')
	parser.add_argument('--seed', required = False, type = int, nargs='+', default = [9999], help='The (list of) seed.')
	parser.add_argument('--num_exons', required = False, type = int, default=2, help = 'number of exon in the gene')
	parser.add_argument('--time_traverse_gene', required=True, type=float, nargs='+', help='The list of time (minutes) to traverse the gene.')
	parser.add_argument('--feat_length', required=True, type=float, nargs='+', help='The list of possible length of a feature in a gene in kb.')
	parser.add_argument('--time', required=False, type=int, default=[5], nargs='+', help='The interval between each labeling milestone.')
	parser.add_argument('--output_folder', required=True, type=str, help='The folder where we will save the results of all the runs.')
	parser.add_argument('--save_fn', required=True, type=str, help='The file where we will save the design dataframe.')
	parser.add_argument('--vary_middle', action='store_true', help='if this flag is present, we will vary the elongation rate in the middle of the gene')
	parser.add_argument('--vary_across', action='store_true', help='if this flag is present, we will vary the elongation rate across the gene')
	parser.add_argument('--vary_sd', required=False, default=0, type=float, help='The standard deviation of the elongation rate. This is only applicable if we are varying the elongation rate ACROSS the gene.')
	parser.add_argument('--vary_bin_bp', required=False, type=int, default=0, help='The bin size for which we will calculate the elongation rates')
	parser.add_argument('--vary_fold', required=False, type=float, default=1, help='The fold change in elongation rate for the variable elongation rate')
	parser.add_argument('--pair_end', action='store_true', help='Whether we are simulating pair end reads or not')
	args = parser.parse_args()
	helper.create_folder_for_file(args.save_fn)
	print ('create_design_parameters.py: Done getting command line argument')
	# design_df = get_constant_elong_design_df(args.lambda_init, args.burst_size, args.insertsize_min, args.insertsize_max, args.read_length, args.seed, args.num_exons, args.feat_length, args.time_traverse_gene, args.output_folder, args.save_fn, args.time)
	if args.vary_middle:
		design_df = get_vary_Middle_elong_design_df(args.lambda_init, args.burst_size, args.insertsize_min, args.insertsize_max, args.read_length, args.seed, args.num_exons, args.feat_length, args.time_traverse_gene, args.output_folder, args.save_fn, args.time, vary_bin_bp=args.vary_bin_bp, vary_fold=args.vary_fold, pair_end=args.pair_end)
	elif args.vary_across:
		design_df = get_vary_Across_elong_design_df(args.lambda_init, args.burst_size, args.insertsize_min, args.insertsize_max, args.read_length, args.seed, args.num_exons, args.feat_length, args.time_traverse_gene, args.output_folder, args.save_fn, args.time, vary_bin_bp=args.vary_bin_bp, vari_sd = args.vary_sd, pair_end=args.pair_end)
	print('Done!')

# on 10/27/2024: python create_design_matrix.py --lambda_init 2 10 30 60 120 --burst_size 15 --time_traverse_gene 3 15 30 75 --feat_length 1 3 6 12  --time 5 --output_folder '/gladstone/engelhardt/lab/hvu//RNA_rates/elong_rates/pair_end_reads/oct_24/constant' --save_fn /gladstone/engelhardt/lab/hvu//RNA_rates/elong_rates/pair_end_reads/oct_24/constant/design_matrix.csv --vary_across --vary_sd 0 --vary_bin_bp 1000 --pair_end
# python create_design_params.py --time_traverse_gene 3 6 12 15 30 60 --feat_length 1.65 3.35 6.75 13.5 27 --output_folder /gladstone/engelhardt/lab/hvu/RNA_rates/elong_rates/no_reads/constant/ --save_fn /gladstone/engelhardt/lab/hvu/RNA_rates/elong_rates/no_reads/constant/design_matrix.csv
#  python create_design_params.py --time_traverse_gene 3 6 12 15 30 60 --feat_length 1.65 3.35 6.75 13.5 27 --output_folder /gladstone/engelhardt/lab/hvu/RNA_rates/elong_rates/pair_end_reads/constant/ --save_fn /gladstone/engelhardt/lab/hvu/RNA_rates/elong_rates/pair_end_reads/constant/design_matrix.csv  --pair_end  --insertsize_min 200 --insertsize_max 300 --read_length 100
# python create_design_params.py --time_traverse_gene 3 6 12 15 30 60 --feat_length 1.65 3.35 6.75 13.5 27 --output_folder /gladstone/engelhardt/lab/hvu/RNA_rates/elong_rates/no_reads/variable/bin0.2_fold2 --save_fn /gladstone/engelhardt/lab/hvu/RNA_rates/elong_rates/no_reads/variable/bin0.2_fold2/design_matrix.csv  --vary_bin_bp 200 --vary_fold 2
# python create_design_params.py --time_traverse_gene 3 6 12 15 30 60 --feat_length 1.65 3.35 6.75 13.5 27 --output_folder /gladstone/engelhardt/lab/hvu/RNA_rates/elong_rates/no_reads/variable/bin0.2_fold4 --save_fn /gladstone/engelhardt/lab/hvu/RNA_rates/elong_rates/no_reads/variable/bin0.2_fold4/design_matrix.csv  --vary_bin_bp 200 --vary_fold 4
# python create_design_params.py --time_traverse_gene 3 6 12 15 30 60 --feat_length 1.65 3.35 6.75 13.5 27 --output_folder /gladstone/engelhardt/lab/hvu/RNA_rates/elong_rates/no_reads/variable/bin1_fold2 --save_fn /gladstone/engelhardt/lab/hvu/RNA_rates/elong_rates/no_reads/variable/bin1_fold2/design_matrix.csv --vary_bin_bp 1000 --vary_fold 2
# python create_design_params.py --time_traverse_gene 3 6 12 15 30 60 --feat_length 1.65 3.35 6.75 13.5 27 --output_folder /gladstone/engelhardt/lab/hvu/RNA_rates/elong_rates/no_reads/variable/bin1_fold4 --save_fn /gladstone/engelhardt/lab/hvu/RNA_rates/elong_rates/no_reads/variable/bin1_fold4/design_matrix.csv --vary_bin_bp 1000 --vary_fold 4
# python create_design_params.py --time_traverse_gene 3 6 12 15 30 60 --feat_length 1.65 3.35 6.75 13.5 27 --output_folder /gladstone/engelhardt/lab/hvu/RNA_rates/elong_rates/no_reads/variable/bin0.5_fold4 --save_fn /gladstone/engelhardt/lab/hvu/RNA_rates/elong_rates/no_reads/variable/bin0.5_fold4/design_matrix.csv --vary_bin_bp 500 --vary_fold 4
# python create_design_params.py --time_traverse_gene 3 6 12 15 30 60 --feat_length 1.65 3.35 6.75 13.5 27 --output_folder /gladstone/engelhardt/lab/hvu/RNA_rates/elong_rates/no_reads/variable/bin0.5_fold2 --save_fn /gladstone/engelhardt/lab/hvu/RNA_rates/elong_rates/no_reads/variable/bin0.5_fold2/design_matrix.csv --vary_bin_bp 500 --vary_fold 2