'''
This code will create the design matrix for simulation for a single gene that we have the gtf file for.
Here, the application is when we have a bunch of gtf files corresponding to a bunch of genes, and we want to simulate the transcription process for a gene (a gtf file) with different settings of gene expression elongation rates, etc. This script create the design matrices for those simulations.
'''
import pandas as pd
import numpy as np
import argparse
import helper
import os
import itertools
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) # this is supposed to let the code know that some of the import path is outside of this folder
import transcription.simulate_multiple_experiments as sim
import elong_analysis.create_design_matrix as design
ONE_KB=1000
def read_org_gtf_df(org_gtf_fn):
	org_gtf_df = pd.read_csv(org_gtf_fn, sep='\t', header=0, index_col=None)
	# if org_gtf_fn is /path/to/ENSG00000000003_ENST00000612152.gtf, then gene_name = ENSG00000000003_ENST00000612152
	gene_name = org_gtf_fn.split('/')[-1].split('.gtf')[0]
	return gene_name, org_gtf_df

def create_custom_elongf_df(org_gtf_df, time_traverse_gene=1, output_folder='./'):
	'''
	This function will create the elongf_df for the gene, figure out the elongation rates based on time_traverse_gene and save it in the output_folder
	Right now we onlly do constant elongation rate, but soon we will have more features
	:param org_gtf_df:
	:param time_traverse_gene:
	:param output_folder:
	:return:
	'''
	# gene end is the end of the last PAS
	gene_start = org_gtf_df['start'].min()
	gene_length = org_gtf_df[org_gtf_df['feature']=='PAS']['end'].max() - gene_start
	constant_elong_rate = gene_length / time_traverse_gene / ONE_KB
	row = pd.Series({'length': np.inf, 'txrate': constant_elong_rate, 'start': gene_start, 'end': np.inf})
	elongf_df = pd.DataFrame(columns=['length', 'txrate', 'start', 'end'])
	elongf_df.loc[0] = row
	elongf_fn = os.path.join(output_folder, 'input_elongf_df.csv')
	elongf_df.to_csv(elongf_fn, sep='\t', header=True, index=False)
	return elongf_df, constant_elong_rate

def get_variable_elong_design_df(org_gtf_fn, lambda_init, burst_size, insertsize_min, insertsize_max, read_length, seed, time_traverse_gene, output_folder, save_fn, label_time,  h_bounds=(0.1, 10), vari_bin_bp=1000, vari_fold=1, pair_end=False):
	gene_name, org_gtf_df = read_org_gtf_df(org_gtf_fn)
	design_df = pd.DataFrame(columns=['lambda_init', 'burst_size', 'insertsize_min', 'insertsize_max', 'read_length', 'seed', 'time_traverse_gene', 'pair_end'])
	entry_dict = {'lambda_init': lambda_init, 'burst_size': burst_size, 'insertsize_min': insertsize_min, 'insertsize_max': insertsize_max, 'read_length': read_length, 'seed': seed, 'time': label_time, 'time_traverse_gene': time_traverse_gene, 'pair_end': [pair_end]}
	rows = design.generate_combinations_dict(entry_dict)  # list of dictionaries, each dictionary can be used as row in design_fn
	for row_idx, row in enumerate(rows):
		design_df.loc[row_idx] = row
	design_df['output_folder'] = design_df.apply(lambda x: os.path.join(output_folder, f'result_{x.name}'), axis=1)  # apply to each row
	for idx, row in design_df.iterrows():
		helper.make_dir(row['output_folder'])
		_, constant_elong_rate = create_custom_elongf_df(org_gtf_df, time_traverse_gene=row['time_traverse_gene'], output_folder=row['output_folder'])
		design_df.loc[idx, 'elongf_fn'] = os.path.join(row['output_folder'], 'input_elongf_df.csv')
		design_df.loc[idx, 'constant_elong_rate'] = constant_elong_rate
		org_gtf_fn_output = os.path.join(row['output_folder'], f'{gene_name}.gtf.gz')
		# soft link the org_gtf_fn to the output folder, this is the solution for now, until we want to change the splicing and cleavage rates
		os.system(f'ln -s {org_gtf_fn} {org_gtf_fn_output}')
		design_df.loc[idx, 'gtf_fn'] = org_gtf_fn_output
		# if the constant_elong_rate is out of the bound we want to test the limits of the elongation rate, we shall skip this by creating dummies files so that snakemake will pick up on missing files
		if constant_elong_rate < h_bounds[0] or constant_elong_rate > h_bounds[1]:
			design.create_dummies_files(row['output_folder'])
	design_df.to_csv(save_fn, sep='\t', header=True, index=False)
	return



if __name__ == '__main__':
	parser = argparse.ArgumentParser(description = "This code will take in a gtf file, and create the design matrix for simulation for a single gene. Each row in the design matrix will correspond to sets of parameters related to gene expression, elongation rates, read generations etc.")
	parser.add_argument('--org_gtf_fn', required = True, type = str, help='orginal gtf file for the gene')
	parser.add_argument('--lambda_init', required = False, type = float, nargs='+', default=[2],  help='Average number of init events per minute')
	parser.add_argument('--burst_size', required = False, type = float, nargs='+', default=[10], help='Average number of new transcripts in each burst')
	parser.add_argument('--insertsize_min', required = False, type = int, nargs='+', default=[-1], help='minimum read length filter')
	parser.add_argument('--insertsize_max', required = False, type = int, nargs='+', default=[-1], help='maximum read length filter')
	parser.add_argument('--read_length', required = False, type = int, nargs='+', default=[-1], help='read length flter')
	parser.add_argument('--seed', required = False, type = int, nargs='+', default = [9999], help='The (list of) seed.')
	parser.add_argument('--time_traverse_gene', required=True, type=float, nargs='+', help='The list of time (minutes) to traverse the gene.')
	parser.add_argument('--label_time', required=False, type=int, default=[5], nargs='+', help='The interval between each labeling milestone.')
	parser.add_argument('--output_folder', required=True, type=str, help='The folder where we will save the results of all the runs.')
	parser.add_argument('--save_fn', required=True, type=str, help='The file where we will save the design dataframe.')
	parser.add_argument('--vari_bin_bp', required=False, type=int, default=0, help='The bin size for which we will calculate the elongation rates')
	parser.add_argument('--vari_fold', required=False, type=float, default=1, help='The fold change in elongation rate for the variable elongation rate')
	parser.add_argument('--pair_end', action='store_true', help='Whether we are simulating pair end reads or not')
	args = parser.parse_args()
	helper.make_dir(args.output_folder)
	helper.create_folder_for_file(args.save_fn)
	print ('create_design_parameters.py: Done getting command line argument')
	# design_df = get_constant_elong_design_df(args.lambda_init, args.burst_size, args.insertsize_min, args.insertsize_max, args.read_length, args.seed, args.num_exons, args.feat_length, args.time_traverse_gene, args.output_folder, args.save_fn, args.time)
	design_df = get_variable_elong_design_df(args.org_gtf_fn, args.lambda_init, args.burst_size, args.insertsize_min, args.insertsize_max, args.read_length, args.seed, args.time_traverse_gene, args.output_folder, args.save_fn, args.label_time, vari_bin_bp=1000, vari_fold=1, pair_end=False)
	print('Done!')

