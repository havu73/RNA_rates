import os
from itertools import product  # fot getting all the combinations of the lengths of the features
os.chdir('/gladstone/engelhardt/home/hvu/source/RNA_rates/splicingrates/simulations')
import pandas as pd
import numpy as np
import seaborn as sns
from transcription.experiment import Experiment
from transcription.transcripts import Transcript
import matplotlib.pyplot as plt
from transcription import simulate_multiple_experiments as sim
plt.style.use('default')
plt.rcParams['figure.facecolor'] = 'white'
import helper
import estimate_splice as estSplice
import visualize_simulations as viz
import pandas as pd
ONE_KB=1000
SEED = 9999
import argparse

def get_gtf_df_list_variable_intronH(intronH_range, elong_h_list, feat_idx_to_vary, length_fold_list=None, default_length:float=1., RTR:int=sim.DEFAULT_SIM_FEAT_LEN, default_intronH:float=5.):
	'''
	tested get_gtf_df_list([1,0.5], [2,3],[1],1))
	This function will create a list of gtf_df for which we will simulate transcripts later.
	:param h_list: each value corresponds to the ground truth elongation rate for each feature.
	:param length_range: each value corresponds to one possible length of a feature.
	:param feat_idx_to_vary: each value correpsonds to one feature for which we will vary the length. The rest of the features will have the default length.
	:param default_length: the default length of a feature (if it is set not to vary).
	:return: a list of gtf_df.
	'''
	# create a gtf_df for each elongation rate
	gtf_df_list = []
	# create list of tuples of all possible combinations of lengths for the features. Each tuple: (length1, length2, length3, etc.). Note we only care about the lengths of the exons, will add introns in the next step just to later get rid of them (to make use of existing functions in simulate_multiple_experiments)
	nExons = int((len(elong_h_list)+1)/2)  # bc the gene is like: exon-intron-exon-intron-exon-intron-exon, etc.
	num_features =len(elong_h_list)
	if length_fold_list==None:
		length_fold_list = [default_length]*num_features
	for vary_intronH in intronH_range:
		intronH_list = [default_intronH] * num_features
		intronH_list[feat_idx_to_vary] = vary_intronH
		gtf_df = sim.create_variable_gtf_df(nExons=nExons, elong_fold_list=elong_h_list, length_fold_list=length_fold_list, SIM_FEAT_LEN=ONE_KB, RTR=RTR, intronH_fold_list=intronH_list)  # all lengths are in input of ONE_KB unit
		# now do the work of filtering out the introns
		gtf_df_list.append(gtf_df)
	return gtf_df_list


def calculate_ratio_before_after_intron_endpoint(gtf_df, coverage_df, startT_idx, endT_idx, bp_around_intron=30):
	"""
	Given the coverage data across the gene, we will calculate the ratio of read coverage before and after the intron endpoint for each intron. We will do so for time-tagged reads, so for each time point we have one statistics.
	:param gtf_df:
	:param coverage_df:
	:param startT_idx:
	:param endT_idx:
	:return: gtf_df with added columns of H0, H1, H2, etc. where H{i} is the ratio of read coverage before and after the intron endpoint at time i
	"""
	intron_endpoints = gtf_df[gtf_df['is_intron']]['end'].values # a list with length equal to the number of introns
	intron_startpoints = gtf_df[gtf_df['is_intron']]['start'].values
	intron_idx_in_gtf = gtf_df[gtf_df['is_intron']].index
	time_idx_list = range(startT_idx, endT_idx+1)
	for time_idx in time_idx_list:
		gtf_df[f'before_H{time_idx}']= float(0)  # set default values of the splice read coverage index
		gtf_df[f'after_H{time_idx}'] = float(0)
		gtf_df[f'before_I{time_idx}'] = float(0)  # at the beginning of the intron
		gtf_df[f'after_I{time_idx}'] = float(0)
	beforeH_idx = intron_endpoints- bp_around_intron
	afterH_idx = intron_endpoints + bp_around_intron
	beforeI_idx= intron_startpoints- bp_around_intron
	afterI_idx = intron_startpoints + bp_around_intron
	for time_idx in time_idx_list:
		for intron_idx, iEndpoint in enumerate(intron_endpoints):
			nom = coverage_df.loc[(beforeH_idx[intron_idx]):(iEndpoint-1), time_idx].mean()
			denom = coverage_df.loc[iEndpoint:(afterH_idx[intron_idx]-1), time_idx].mean()
			gtf_df.loc[intron_idx_in_gtf[intron_idx], f'before_H{time_idx}'] = nom
			gtf_df.loc[intron_idx_in_gtf[intron_idx], f'after_H{time_idx}'] = denom
			nom = coverage_df.loc[(beforeI_idx[intron_idx]):(intron_startpoints[intron_idx]-1), time_idx].mean()
			denom = coverage_df.loc[intron_startpoints[intron_idx]:(afterI_idx[intron_idx]-1), time_idx].mean()
			gtf_df.loc[intron_idx_in_gtf[intron_idx], f'before_I{time_idx}'] = nom
			gtf_df.loc[intron_idx_in_gtf[intron_idx], f'after_I{time_idx}'] = denom
	gtf_df = gtf_df[['feature', 'start', 'end', 'length', 'txrate', 'is_intron', 'intron_h'] +
					[f'before_H{i}' for i in time_idx_list] +
					[f'after_H{i}' for i in time_idx_list] +
					[f'before_I{i}' for i in time_idx_list] +
					[f'after_I{i}' for i in time_idx_list]]
	return gtf_df

def calculate_spliceH_given_coverage_df(coverage_df, gtf_df, endT_idx=2):
	# for each intron, get the intron end, and calculate the estimated splice half-life using functions in estimate_splice.py
	intron_endpoints = gtf_df[gtf_df['is_intron']]['end'].values
	intron_idx_in_gtf = gtf_df[gtf_df['is_intron']].index
	for intron_idx, iEndpoint in enumerate(intron_endpoints):
		h_dict = estSplice.calculate_splice_given_coverage_df(coverage_df, iEndpoint, gtf_df, endT_idx=endT_idx)
		gtf_df.loc[intron_idx_in_gtf[intron_idx], h_dict.keys()] = pd.Series(h_dict)
	return gtf_df

def run_one_experiment(gtf_df, PDB, time=5, seed:int=SEED, burst_size=10, lambda_init=2):
	'''
	tested result_df = run_one_experiment(gtf_df_list[0], True)
	This function will simulate one experiment, calculate the elongation rates for each feature, and return the results in the form of a dataframe.
	:param gtf_df:
	:param label_time:
	:param num_timepoints:
	:param degrade_rate:
	:param intron_h:
	:param PAS_h:
	:param RTR:
	:param lambda_init:
	:param burst_size:
	:param wiggle_room:
	:param simulate_cleavage:
	:param PDB:
	:param max_time_for_equilibrium:
	:param save_folder
	:return:
	'''
	# set the random seed
	np.random.seed(seed)
	label_time = np.arange(0,4) if PDB else np.arange(0,3)
	label_time = label_time * time # each label_time is <time> minute apart
	wiggle_room = 0.3  # the wiggle room for the burst event. See the comments in function Experiment.init_bursting_transcripts to understand the meaning of this parameter. If not sure, don't modify it.
	# if I want to simulate situation such that there is no read being generated from fragments (only the whole transcripts are sequenced), the following parameters should be set carefully:
	simulate_cleavage = False  ## for this problem, we don't need to simulate cleavage because we really only care about calculating the elongation speed of the transcripts. We skip splicing and cleavage for now.
	max_time_for_equilibrium = 50
	save_folder = None
	eta_val = helper.DFT_ETA_VALUE
	insertsize_min = -1 #200
	insertsize_max = -1 #300
	read_length = -1 #150
	# if I set the read values to -1, the program will just generate fragments and not get rid of any portion of the fragments.
	Transcript.set_gtf_df(gtf_df)
	Transcript.set_read_params(eta_val=eta_val, insertsize_min=insertsize_min, insertsize_max=insertsize_max, read_length=read_length)
	Experiment.set_gtf_df(gtf_df)
	Experiment.set_simulate_cleavage(simulate_cleavage)
	Experiment.set_read_params(eta_val=eta_val, insertsize_min=insertsize_min, insertsize_max=insertsize_max, read_length=read_length)
	Experiment.set_burst_init_params(lambda_init=lambda_init, burst_size=burst_size, wiggle_room=wiggle_room)
	exp_list = sim.generate_exp_given_one_gtf(gtf_df, save_folder=save_folder, label_time=label_time,lambda_init=lambda_init, burst_size=burst_size, wiggle_room=wiggle_room, eta_val=eta_val, insertsize_min=insertsize_min, insertsize_max=insertsize_max, read_length=read_length,simulate_cleavage=simulate_cleavage, PDB=PDB, max_time_for_equilibrium=max_time_for_equilibrium)
	endpoint_df = viz.get_endpoints_across_time(exp_list)
	coverage_df = viz.count_timeDep_read_coverage(exp_list[-1], endpoint_df, N=1, num_timepoints=len(exp_list))
	# now calculate the elongation rates for each feature
	startT_idx = 1 if PDB else 0
	endT_idx = len(label_time)-1
	# try:
		# result_df = calculate_ratio_before_after_intron_endpoint(gtf_df, coverage_df, startT_idx, endT_idx)
	result_df = calculate_spliceH_given_coverage_df(coverage_df, gtf_df, endT_idx=endT_idx)
	# except:
	# 	print('Failed to calculate elongation rates for this gtf_df. We will return an empty dataframe.  This should never happen though. Check the code!')
	# 	result_df = pd.DataFrame()
	# 	print(gtf_df)
	print('Done with one experiment!')
	print(result_df.head())
	return result_df




if __name__ == '__main__':
	parser = argparse.ArgumentParser(description = "This code will take in the ground truth values of splicing rates for introns in the genes, and it will simulate the transcription process, and return the ratio of read coverage before and after the introns endpoint.")
	parser.add_argument('--PDB', action='store_true' , help='Whether we simulate the transcription with PDB or not')
	parser.add_argument('--vary_intronH_list', required = True, type = float, nargs='+', help='different values of intronH that we will use to simulate the transcription process.')
	parser.add_argument('--elong_fold_list', required=True, type=float, nargs='+', help='The list of elongation rates for the features.')
	parser.add_argument('--length_fold_list', required=True, type=float, nargs='+', help='The list of lengths for the features. If not provided, we will use the default_length for all features.')
	parser.add_argument('--feat_idx_to_vary', required=False, type=int, default= 1, help = 'indices of features in the genes for which we will vary the intronH values. The rest of the features will have the default length and intronH. 0 corresponds to the first EXON in the gene')
	parser.add_argument('--output_fn', required=True, type=str, help = 'Where we will save the results.')
	parser.add_argument('--time', required=False, type=int, default=5, help='The interval between each labeling milestone.')
	parser.add_argument('--burst_size', required = False, type = int, default=10, help='Average number of new transcripts in each burst')
	parser.add_argument('--lambda_init', required = False, type = float, default=2, help='average number of burst events per minute')
	parser.add_argument('--seed', required=False, type=int, help='The seed.')
	parser.add_argument('--insertsize_min', required=False, type=int, default=-1, help='The minimum insert size of the reads.')
	parser.add_argument('--insertsize_max', required=False, type=int, default=-1, help='The maximum insert size of the reads.')
	parser.add_argument('--read_length', required=False, type=int, default=-1, help='The length of the reads.')
	args = parser.parse_args()
	helper.create_folder_for_file(args.output_fn)
	print ('evaluate_splicing.py: Done getting command line argument')
	RTR = helper.DEFAULT_SIM_FEAT_LEN
	gtf_df_list = get_gtf_df_list_variable_intronH(args.vary_intronH_list, args.elong_fold_list, feat_idx_to_vary = args.feat_idx_to_vary, length_fold_list=args.length_fold_list, RTR=sim.DEFAULT_SIM_FEAT_LEN, default_intronH=5.)
	print('Done creating gtf_df_list! We will run {} experiments.'.format(len(gtf_df_list)))
	# apply function to run experiment and calculate the results of elongation rates for each feature
	result_df_list = list(map(lambda x: run_one_experiment(gtf_df_list[x], args.PDB, time=args.time, seed=args.seed, burst_size=args.burst_size, lambda_init=args.lambda_init), range(len(gtf_df_list))))
	# save the results into save_fn
	result_df = pd.concat(result_df_list, ignore_index=True)
	result_df.to_csv(args.output_fn, index=False, header=True, sep=',')
	print('Done!')
