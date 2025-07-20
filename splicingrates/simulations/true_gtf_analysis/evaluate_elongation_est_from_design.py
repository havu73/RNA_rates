import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) # this is supposed to let the code know that some of the import path is outside of this folder
import numpy as np
from transcription import simulate_multiple_experiments as sim
from transcription import from_reads_to_coverage as read2cov
import helper
import visualize_simulations as viz
from estimates.elongation import estElong

import pandas as pd
ONE_KB=1000
SEED = 9999
import argparse
# print working directory
print(os.getcwd())

def get_gtf_df(gtf_fn):
	'''
	This function will read the gtf_fn
	:param gtf_fn:
	:return: gtf_df for the gene
	'''
	gtf_df = pd.read_csv(gtf_fn, header=0, index_col=None, sep = '\t')
	return gtf_df


def get_one_output_df(gtf_df, pred_h, predH_mean, nFeatures):
	'''
	tested result_df = get_one_output_df(gtf_df_list[0],  np.array([[999, 499], [1001, 501]]), np.array([1000, 500]), 2)
	This function will create an output_df that show the results of the simulation and the predicted elongation rates.
	:param gtf_df:
	:param pred_h:
	:param nExons:
	:return:
	'''
	result_df = gtf_df.reset_index(drop=True)
	# we only care about the prediction of features in the gene
	pred_h = pred_h[:, :nFeatures]
	predH_mean = predH_mean[:nFeatures]
	# now, let's report the results
	result_df = result_df.loc[:nFeatures-1, :] # we only want to keep the rows related to features of the gene
	# first add the result of the average elongation rate
	result_df['elong_h'] = predH_mean  # the average elongation rate
	print('result_df.head():')
	print(result_df.head())
	return result_df


def read_coverage_df(coverage_fn):
	coverage_df = pd.read_csv(coverage_fn, header=0, index_col=None, sep = '\t', compression='gzip')
	# if columns of coverage_df is '0' or '1', '2', '3', change it to integer
	curr_columns = list(coverage_df.columns.copy())
	for idx, col in enumerate(curr_columns):
		try:
			col = int(col)
			curr_columns[idx] = col
		except:
			pass
	coverage_df.columns = curr_columns
	return coverage_df

def run_one_experiment(gtf_df, output_folder, elongf_df=None, PDB=False, time=5, seed:int=SEED, burst_size=10, lambda_init=2, h_bin_bp_list=[], resimulate=False, dont_estimate=False, pair_end=False):
	# set the random seed
	np.random.seed(seed)
	label_time = np.arange(0,4) if PDB else np.arange(0,3)
	label_time = label_time * time # each label_time is <time> minute apart
	wiggle_room = 0.3
	simulate_cleavage = False  ## for this problem, we don't need to simulate cleavage because we really only care about calculating the elongation speed of the transcripts. We skip splicing and cleavage for now.
	max_time_for_equilibrium = 50
	save_folder = output_folder
	eta_val = helper.DFT_ETA_VALUE
	insertsize_min = 200 if pair_end else -1 # if we simulate reads, we will set the insert size to be 200. Otherwise right now keep it simple by saying we don't simulate reads.
	insertsize_max = 300 if pair_end else -1 # if we simulate reads, we will set the insert size to be 500. Otherwise right now keep it simple by saying we don't simulate reads.
	read_length = 100 if pair_end else -1 # if we simulate reads, we will set the read length to be 100. Otherwise right now keep it simple by saying we don't simulate reads.
	exp_list = sim.generate_exp_given_one_gtf(gtf_df, elongf_df=elongf_df, save_folder=save_folder, label_time=label_time,target_exp=5, num_total_transcript_millions=100, burst_size=burst_size, wiggle_room=wiggle_room, lambda_init=lambda_init,eta_val = eta_val, insertsize_min = insertsize_min, insertsize_max = insertsize_max, read_length = read_length,simulate_cleavage = simulate_cleavage, PDB = PDB, max_time_for_equilibrium = max_time_for_equilibrium, pair_end=pair_end)
	endpoint_df = read2cov.get_endpoints_across_time(exp_list)  # for each transcript, figure out where the endpoint at each time point is
	exp = exp_list[-1]
	exp.get_reads_df()
	endpoint_df = read2cov.get_endpoints_across_time(exp_list)  # for each transcript, figure out where the endpoint at each time point is
	exp.tag_reads_by_timepoint(endpoint_df)
	import pdb; pdb.set_trace()

def check_elong_params(elongf_fn=None):
	if not os.path.isfile(elongf_fn):
		raise ValueError('The elongf_fn does not exist.')
	elongf_df = pd.read_csv(elongf_fn, header=0, index_col=None, sep = '\t')
	return elongf_df

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description = "This code will take in the ground truth values of splicing rates for introns in the genes, and it will simulate the transcription process, and return the ratio of read coverage before and after the introns endpoint.")
	parser.add_argument('--PDB', action='store_true' , help='Whether we simulate the transcription with PDB or not')
	parser.add_argument('--gtf_fn', required=True, type=str, help='The list of lengths for the features. If not provided, we will use the default_length for all features.')
	parser.add_argument('--elongf_fn', required=True, type=str, default=None, help='Where we have the elongation rates')
	parser.add_argument('--output_folder', required=True, type=str, help = 'Where we will save the results.')
	parser.add_argument('--h_bin_bp_list', required=False, default=[200, 300, 400, 500, 600, 700, 800, 900, 1000, 1250, 1500, 1750, 2000], nargs='+', type = int, help='The bin size for which we will calculate the elongation rates')
	parser.add_argument('--time', required=False, type=int, default=5, help='The interval between each labeling milestone.')
	parser.add_argument('--burst_size', required = False, type = int, default=10, help='Average number of new transcripts in each burst')
	parser.add_argument('--lambda_init', required = False, type = float, default=2, help='average number of burst events per minute')
	parser.add_argument('--seed', required=False, type=int, help='The seed.')
	parser.add_argument('--insertsize_min', required=False, type=int, default=-1, help='The minimum insert size of the reads.')
	parser.add_argument('--insertsize_max', required=False, type=int, default=-1, help='The maximum insert size of the reads.')
	parser.add_argument('--read_length', required=False, type=int, default=-1, help='The length of the reads.')
	parser.add_argument('--dont_estimate', action='store_true', help='If we do not want to estimate the elongation rates.)')
	parser.add_argument('--resimulate', action='store_true', help='If we want to resimulate the data.')
	parser.add_argument('--pair_end', action='store_true', help='If we want to simulate pair end reads.')
	args, unknown = parser.parse_known_args()
	helper.make_dir(args.output_folder)
	print ('evaluate_splicing.py: Done getting command line argument')
	RTR = helper.DEFAULT_SIM_FEAT_LEN
	elongf_df = check_elong_params(args.elongf_fn)
	gtf_df = get_gtf_df(args.gtf_fn)
	gtf_df, elongf_df = sim.align_gtf_df_with_elongf_df(gtf_df, elongf_df)
	print(gtf_df)
	print(elongf_df)
	print('Done creating gtf_df and elongf_df')
	run_one_experiment(gtf_df, args.output_folder, elongf_df=elongf_df, PDB=False, time=5, seed=SEED, burst_size=args.burst_size, lambda_init=args.lambda_init, h_bin_bp_list=args.h_bin_bp_list, dont_estimate=args.dont_estimate, pair_end=args.pair_end, resimulate=args.resimulate)
	# save the results into save_fn
	print('Done!')

# python evaluate_elongation_est_from_desing.py --lambda_init 2 --burst_size 10 --insertsize_min 200 --insertsize_max 300 --read_length 100 --seed 9999 --constant_elong_rate nan --length_fold_list 1.65 1.65 1.65 --time_traverse_gene 15 --pair_end --output_folder /gladstone/engelhardt/lab/hvu/RNA_rates/elong_rates/pair_end_reads/constant/result_3 --elongf_fn /gladstone/engelhardt/lab/hvu/RNA_rates/elong_rates/pair_end_reads/constant/result_3/input_elongf_df.csv --avg_elong_rate 0.33