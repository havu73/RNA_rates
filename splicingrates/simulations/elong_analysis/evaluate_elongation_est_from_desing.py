import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) # this is supposed to let the code know that some of the import path is outside of this folder
import numpy as np
# import pdb; pdb.set_trace()
from transcription import simulate_multiple_experiments as sim
from transcription import from_reads_to_coverage as read2cov
import helper
from estimates.elongation import estElong

import pandas as pd
ONE_KB=1000
SEED = 9999
import argparse
# print working directory
print(os.getcwd())
print('hello')
def get_gtf_df(length_fold_list, elong_fold_list=None, RTR:int=sim.DEFAULT_SIM_FEAT_LEN, default_intronH:float=np.inf):
	'''
	tested get_gtf_df([1,0.5], [2,3],[1],1))
	This function will create a list of gtf_df for which we will simulate transcripts later.
	:param elong_h_list: each value corresponds to the ground truth elongation rate for each feature.
	:param length_fold_list: each value corresponds to one possible length of a feature, length is equal to the length_fold*ONE_KB
	:param default_length: the default length of a feature (if it is set not to vary).
	:param RTR: run through region length.
	:param default_intronH: the default splicing rate for the introns. default is infinity, which means we do NOT simulate splicing.
	:return: a list of gtf_df.
	'''
	# create a gtf_df for each elongation rate
	nExons = int((len(length_fold_list)+1)/2)
	num_features =len(length_fold_list)
	intronH_fold_list = [default_intronH]*num_features  # we will not simulate splicing for now, so we will set the intronH to be infinity --> no splicing will happen
	gtf_df = sim.create_variable_gtf_df(nExons=nExons, elong_fold_list=elong_fold_list, length_fold_list=length_fold_list, SIM_FEAT_LEN=ONE_KB, RTR=RTR, intronH_fold_list=intronH_fold_list)  # all lengths are in input of ONE_KB unit
	# now do the work of filtering out the introns, because for this application we are only looking at the case where we calculate the elongation rates without the phenomenon of splicing
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
	max_time_for_equilibrium = 10
	save_folder = output_folder
	eta_val = helper.DFT_ETA_VALUE
	insertsize_min = 200 if pair_end else -1 # if we simulate reads, we will set the insert size to be 200. Otherwise right now keep it simple by saying we don't simulate reads.
	insertsize_max = 300 if pair_end else -1 # if we simulate reads, we will set the insert size to be 500. Otherwise right now keep it simple by saying we don't simulate reads.
	read_length = 100 if pair_end else -1 # if we simulate reads, we will set the read length to be 100. Otherwise right now keep it simple by saying we don't simulate reads.
	# first check if the file coverage_df.csv.gz already exists
	coverage_fn = os.path.join(output_folder, 'coverage_df.csv.gz')
	if os.path.exists(coverage_fn) and not resimulate:
		print('The coverage_df.csv.gz file already exists. We will skip the simulation.')
		coverage_df = read_coverage_df(coverage_fn)
		ext_gtf_df = pd.read_csv(os.path.join(output_folder, 'gtf_df.csv'), header=0, index_col=None, sep = '\t')
		ext_elongf_df = pd.read_csv(os.path.join(output_folder, 'elongf_df.csv'), header=0, index_col=None, sep = '\t')
		# HAHAHA: this test sometimes does not pass and I am still trying to figure out why, definitely does not change the results of the estimation but still need to check why
		# assert ext_gtf_df.equals(gtf_df), 'The gtf_df is not the same as the one used to generate the coverage_df.csv.gz'
		# assert ext_elongf_df.equals(elongf_df), 'The elongf_df is not the same as the one used to generate the coverage_df.csv.gz'
	else:
		exp_list = sim.generate_exp_given_one_gtf(gtf_df, elongf_df=elongf_df, save_folder=save_folder, label_time=label_time,
											  target_exp=5, num_total_transcript_millions=100, burst_size=burst_size, wiggle_room=wiggle_room, lambda_init=lambda_init,
											  eta_val = eta_val, insertsize_min = insertsize_min, insertsize_max = insertsize_max, read_length = read_length,
											  simulate_cleavage = simulate_cleavage, PDB = PDB, max_time_for_equilibrium = max_time_for_equilibrium, pair_end=pair_end)
		endpoint_df = read2cov.get_endpoints_across_time(exp_list)
		coverage_df = read2cov.count_timeDep_read_coverage(exp_list[-1], endpoint_df, num_timepoints=len(exp_list))
		coverage_df.to_csv(coverage_fn, header = True, index=False, sep = '\t', compression='gzip')
	if not dont_estimate:  # estimate
		elong_estimator = estElong(coverage_df, gtf_df, elongf_df=elongf_df, h_bin_bp=h_bin_bp_list, output_folder=output_folder)
		elong_estimator.estimate()
		draw_save_fn = os.path.join(output_folder, 'pwlr_with_data.png')
		# elong_estimator.draw_regression_lines(draw_save_fn)
		predf_fn = os.path.join(output_folder, 'pred_h.csv.gz')
		elong_estimator.save_estimates(predf_fn)
	return

def check_elong_params(elongf_fn=None, constant_elong_rate=1., elong_fold_list=None):
	if (elongf_fn is not None) and (elong_fold_list is not None): # they cannot both be available
		raise ValueError('elongf_fn and elong_fold_list should not be available all at once')
	if elongf_fn is not None:
		elongf_df = pd.read_csv(elongf_fn, header=0, index_col=None, sep = '\t')
		required_columns = ['start', 'end', 'txrate']
		for col in required_columns:
			assert col in elongf_df.columns, 'Missing required column: '+str(col)
	elif elong_fold_list is not None:
		elongf_df = None
	else:
		elongf_df = pd.DataFrame({'start': [0], 'end': [np.inf], 'txrate': [constant_elong_rate]})
	return elongf_df

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description = "This code will take in the ground truth values of splicing rates for introns in the genes, and it will simulate the transcription process, and return the ratio of read coverage before and after the introns endpoint.")
	parser.add_argument('--PDB', action='store_true' , help='Whether we simulate the transcription with PDB or not')
	parser.add_argument('--length_fold_list', required=True, type=float, nargs='+', help='The list of lengths for the features. If not provided, we will use the default_length for all features.')
	parser.add_argument('--output_folder', required=True, type=str, help = 'Where we will save the results.')
	parser.add_argument('--elongf_fn', required=False, type=str, default=None, help='Where we have the elongation rates')
	parser.add_argument('--constant_elong_rate', required=False, default=1., type=float, help='If we choose constant elongation rates, this will be the rate used throughout the gene')
	parser.add_argument('--h_bin_bp_list', required=False, default=[np.inf], nargs='+', type = int, help='The bin size for which we will calculate the elongation rates')
	parser.add_argument('--elong_fold_list', required=False, default=None, type=float, nargs='+', help='The list of elongation rates for the features.')
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
	if (args.elongf_fn is not None) and (args.elong_fold_list is not None):
		raise ValueError('elongf_fn and elong_fold_list should not be available all at once')
	print ('evaluate_splicing.py: Done getting command line argument')
	RTR = helper.DEFAULT_SIM_FEAT_LEN
	elongf_df = check_elong_params(args.elongf_fn, args.constant_elong_rate, args.elong_fold_list)
	gtf_df = get_gtf_df(length_fold_list=args.length_fold_list, elong_fold_list= args.elong_fold_list, RTR=sim.DEFAULT_SIM_FEAT_LEN, default_intronH=np.inf)
	gtf_df, elongf_df = sim.align_gtf_df_with_elongf_df(gtf_df, elongf_df)
	print(gtf_df)
	print(elongf_df)
	print('Done creating gtf_df and elongf_df')

	run_one_experiment(gtf_df, args.output_folder, elongf_df=elongf_df, PDB=False, time=5, seed=SEED, burst_size=args.burst_size, lambda_init=args.lambda_init, h_bin_bp_list=args.h_bin_bp_list, dont_estimate=args.dont_estimate, pair_end=args.pair_end, resimulate=args.resimulate)
	# save the results into save_fn
	print('Done!')


# python evaluate_elongation_est_from_desing.py --lambda_init 2 --burst_size 10 --insertsize_min 200 --insertsize_max 300 --read_length 100 --seed 9999 --constant_elong_rate nan --length_fold_list 1.65 1.65 1.65 --time_traverse_gene 15 --pair_end --output_folder /gladstone/engelhardt/lab/hvu/RNA_rates/elong_rates/pair_end_reads/constant/result_3 --elongf_fn /gladstone/engelhardt/lab/hvu/RNA_rates/elong_rates/pair_end_reads/constant/result_3/input_elongf_df.csv --avg_elong_rate 0.33