import pandas as pd
import numpy as np
import argparse
import helper
import os
import itertools

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

def get_design_df(lambda_init, burst_size, insertsize_min, insertsize_max, read_length, seed, num_exons, exon_elong, intron_elong, exon_length, intron_length, output_folder, save_fn, time):
	'''
	:param lambda_init: list of average number of init events per minute
	:param burst_size: list of average number of new transcripts in each burst
	:param insertsize_min: list of minimum read length filter
	:param insertsize_max: list of maximum read length filter
	:param read_length: list of read length filter
	:param seed: list of seed
	:param num_exons: number of exon in the gene
	:param feat_idx_to_vary: indices of features in the genes for which we will vary the elong rate and length.
	:param exon_elong: list of possible elongation rates for exons.
	:param intron_elong: list of possible elongation rates for introns.
	:param exon_length: list of possible lengths for exons
	:param intron_length: list of possible lengths for introns
	:return: a dataframe that contains all the possible combinations of the parameters
	'''
	design_df = pd.DataFrame(columns=['lambda_init', 'burst_size', 'insertsize_min', 'insertsize_max', 'read_length', 'seed', 'elong_fold_list', 'length_fold_list'])
	# find the different combinations of length and elongation rate for exons and introns
	elong_option_list = []
	length_option_list = []
	for exon_idx in range(num_exons-1):
		elong_option_list.append(exon_elong)
		length_option_list.append(exon_length)
		elong_option_list.append(intron_elong)
		length_option_list.append(intron_length)
	elong_option_list.append(exon_elong)
	length_option_list.append(exon_length)
	elong_comb_list = generate_combinations(elong_option_list)  # each element is a tuple of elongation rates for each feature
	length_comb_list = generate_combinations(length_option_list)  # each element is a tuple of lengths for each feature
	entry_dict = {'lambda_init': lambda_init, 'burst_size': burst_size, 'insertsize_min': insertsize_min, 'insertsize_max': insertsize_max, 'read_length': read_length, 'seed': seed, 'elong_fold_list': elong_comb_list, 'length_fold_list': length_comb_list, 'time': time}
	rows = generate_combinations_dict(entry_dict)  # list of dictionaries, each dictionary can be used as row in design_fn
	for row_idx, row in enumerate(rows):
		design_df.loc[row_idx] = row
	design_df['output_fn'] = design_df.apply(lambda x: os.path.join(output_folder, f'result_{x.name}.csv'), axis=1)  # apply to each row
	design_df.to_csv(save_fn, header =True, index = False, sep = '\t')
	return design_df


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description = "This code will take create dataframes that tell us the different parameter combinations that we will use to simulate the transcription process, and where we will find the results file for each of the parameter combinations.")
	parser.add_argument('--PDB', action='store_true' , help='Whether we simulate the transcription with PDB or not')
	parser.add_argument('--lambda_init', required = False, type = float, nargs='+', default=[2],  help='Average number of init events per minute')
	parser.add_argument('--burst_size', required = False, type = float, nargs='+', default=[10, 15, 20], help='Average number of new transcripts in each burst')
	parser.add_argument('--insertsize_min', required = False, type = int, nargs='+', default=[-1], help='minimum read length filter')
	parser.add_argument('--insertsize_max', required = False, type = int, nargs='+', default=[-1], help='maximum read length filter')
	parser.add_argument('--read_length', required = False, type = int, nargs='+', default=[-1], help='read length flter')
	parser.add_argument('--seed', required = False, type = int, nargs='+', default = [9999], help='The (list of) seed.')
	parser.add_argument('--num_exons', required = False, type = int, default=2, help = 'number of exon in the gene')
	parser.add_argument('--feat_idx_to_vary', required=False, type=int, default= [0,1,2], nargs='+', help = 'indices of features in the genes for which we will vary the elong rate and length.')
	parser.add_argument('--exon_elong', required=True, type=float, nargs='+', help='The list of possible elongation rates for exons.')
	parser.add_argument('--intron_elong', required=True, type=float, nargs='+', help='The list of possible elongation rates for introns.')
	parser.add_argument('--exon_length', required=True, type=float, nargs='+', help='The list of possible lengths for exons')
	parser.add_argument('--intron_length', required=True, type=float, nargs='+', help='The list of possible lengths for introns')
	parser.add_argument('--time', required=False, type=int, default=[5], nargs='+', help='The interval between each labeling milestone.')
	parser.add_argument('--output_folder', required=True, type=str, help='The folder where we will save the results of all the runs.')
	parser.add_argument('--save_fn', required=True, type=str, help='The file where we will save the design dataframe.')
	args = parser.parse_args()
	helper.create_folder_for_file(args.save_fn)
	print ('create_design_parameters.py: Done getting command line argument')
	design_df = get_design_df(args.lambda_init, args.burst_size, args.insertsize_min, args.insertsize_max, args.read_length, args.seed, args.num_exons, args.exon_elong, args.intron_elong, args.exon_length, args.intron_length, args.output_folder, args.save_fn, args.time)
	print('Done!')
