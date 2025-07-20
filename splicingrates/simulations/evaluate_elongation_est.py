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
import estimate_elongation as estElong
import visualize_simulations as viz
import pandas as pd
ONE_KB=1000
SEED = 9999
np.random.seed(SEED)
import argparse
def get_whole_gene_ground_truth_values(exon_values):
	'''
	Tested!
	This function will take in the values of elongation rates/ lengths for exons, and return the values of elongation rates for the whole gene with 1 intron in between each exon. We have this function because we want to create a gtf_df for the whole gene (exon, intron, exon, intron, exon,etc.) model using the functions we already created. Right now, we will only care about the elongation rates of the exons, and we will eventually ignore introns, but we create it here to make use of existing functions in simulate_multiple_experiments. We will set the elongation rates of the introns to be 1.
	:param exon_values:
	:return:
	'''
	if len(exon_values) == 1:
		return exon_values
	else:
		result = []
		for exon in exon_values[:-1]:
			result.append(exon)
			result.append(1)
		result.append(exon_values[-1]) # last exon
		return result

def all_length_combinations(length_range, nExons, feat_idx_to_vary, default_length):
	'''
	Tested! all_length_combinations([2,3], 2, [1], 1)
	This function will create all the different combinations of lengths for the features, given that we have nExons features, and we want to vary the length of feat_idx_to_vary features. The rest of the features will have the default length.
	:param length_range:
	:param nExons:
	:param feat_idx_to_vary:
	:param default_length:
	:return: list of tuples, each tuple corresponds to the values of lengths for all exons in the gene.
	'''
	if len(feat_idx_to_vary) == 0:
		return [np.array([default_length]*nExons)]
	possible_value_list = [[default_length]]*nExons # list of lists, each corresponds to possible values of length for each feature
	for idx in feat_idx_to_vary:
		possible_value_list[idx] = length_range
	result = list(product(*possible_value_list))
	return result

def get_gtf_df_list(exon_h_list, length_range, feat_idx_to_vary, default_length, RTR:int=sim.DEFAULT_SIM_FEAT_LEN):
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
	nExons = len(exon_h_list)
	length_tuple_list = all_length_combinations(length_range, nExons, feat_idx_to_vary, default_length) # list of tuple
	length_tuple_list = list(map(get_whole_gene_ground_truth_values, length_tuple_list)) # each tuple corresponds to the values of lengths for all exons AND intron in the gene.
	# create the one ground truth elongation rate for the whole gene (intron and exons)
	gene_h_list = get_whole_gene_ground_truth_values(exon_h_list) # between every pair of exon is an intron with elongation rate of 1 (we will get rid of this line later, we do this right now to make use the functions that we already have)
	# now we will create all the different combinations of lengths for the features
	for length_tuple in length_tuple_list:
		gtf_df = sim.create_variable_gtf_df(nExons=nExons, elong_fold_list=gene_h_list, length_fold_list=length_tuple, SIM_FEAT_LEN=ONE_KB, RTR=RTR)  # all lengths are in input of ONE_KB unit
		# now do the work of filtering out the introns
		gtf_df = gtf_df[gtf_df['is_intron'] == False]
		gtf_df['start'] = gtf_df.loc[0, 'start'] + np.cumsum(gtf_df['length']) - gtf_df['length']
		gtf_df['end'] = gtf_df['start'] + gtf_df['length']
		gtf_df['time'] = (gtf_df['end'] - gtf_df['start']) / gtf_df['txrate'] / ONE_KB  # time to traverse the feature
		gtf_df_list.append(gtf_df)
	return gtf_df_list

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
	result_df['pred_h'] = predH_mean  # the average elongation rate
	# now add the result of the elongation rate calculated in each iteration
	# repeat the result_df nIter times
	nIter = pred_h.shape[0] # number of iterations
	add_df = pd.concat([result_df]*nIter, ignore_index=True)  # first is the average, then the results from each iteration
	add_df['pred_h'] = pred_h.reshape(nIter * nFeatures,)
	# now add the add_df to the result_df
	result_df = pd.concat([result_df, add_df], ignore_index=True)
	# now convert the values of pred_h to ONE_KB unit
	result_df['pred_h'] = result_df['pred_h'] / ONE_KB # from bp/minute to kb/minute
	gene_start = result_df.loc[0, 'start']
	result_df['Udist'] = result_df['start'] - gene_start # distance from the begining of the gene to the start of the feature'
	result_df.rename(columns={'txrate':'true_h'}, inplace=True)
	# now add annotation that the first nExons are average, while the rest are results from each iteration
	result_df['result_type'] = ['average'] * nFeatures + ['oneIter'] * (nIter * nFeatures)
	result_df = result_df[['feature', 'length', 'true_h', 'pred_h', 'Udist', 'result_type']]
	print('result_df.head():')
	print(result_df.head())
	return result_df

def run_one_experiment(gtf_df, PDB, fail_fn, time=5):
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
	label_time = np.arange(0,4) if PDB else np.arange(0,3)
	label_time = label_time * time # each label_time is <time> minute apart
	lambda_init = 2  # average number of burst events per minute (burst event is the event where a burst_size transcripts are created around the same time)
	burst_size = 10  # number of transcripts created in a burst event
	wiggle_room = 0.3  # the wiggle room for the burst event. See the comments in function Experiment.init_bursting_transcripts to understand the meaning of this parameter. If not sure, don't modify it.
	# if I want to simulate situation such that there is no read being generated from fragments (only the whole transcripts are sequenced), the following parameters should be set carefully:
	simulate_cleavage = False  ## for this problem, we don't need to simulate cleavage because we really only care about calculating the elongation speed of the transcripts. We skip splicing and cleavage for now.
	max_time_for_equilibrium = 50
	save_folder = None
	eta_val = helper.DFT_ETA_VALUE
	insertsize_min = -1 #200
	insertsize_max = -1 #300
	read_length = -1 #150 # -1 means we take the whole transcript as the read length.
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
	endT_idx = len(label_time) - 1
	nExons = gtf_df.shape[0] - 2 # the last two rows are the PAS and RTR
	# px0, py0, px1, py1, x0, x1 = estimate_endpoints_acrossTime(coverage_df, startT_idx=startT_idx, endT_idx=endT_idx, time=5, max_segments=30)
	try:
		h, h_mean = estElong.calculate_h_from_coverage(coverage_df, gtf_df, startT_idx=startT_idx, endT_idx=endT_idx, time=time, max_segments=15, trim_N=100, trim_endGene=True, subsample_frac=0.3, SEED=9999, convert_to_KB=True, gapN=1, nIter=10)
		# h: (nIter, nExons), h_mean: (nExons,)
		result_df = get_one_output_df(gtf_df, h, h_mean, nExons)
	except:
		print('Failed to calculate elongation rates for this gtf_df. We will return an empty dataframe.')
		gtf_df.to_csv(fail_fn, index=False, header=True, sep=',')
		result_df = pd.DataFrame(columns = ['feature', 'length', 'true_h', 'pred_h', 'Udist', 'result_type'])
	print('Done with one experiment!')
	return result_df



def create_fail_folder(output_fn):
	'''
	Tested!
	This function will create the folder for the output_fn if it does not exist.
	:param output_fn:
	:return:
	'''
	dirname = os.path.dirname(output_fn)
	raw_fn = os.path.basename(output_fn).split('.csv')
	fail_folder = os.path.join(dirname, raw_fn[0] + '_fail')
	helper.make_dir(fail_folder)
	return fail_folder

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description = "This code will take in the ground truth values of elongation rates for each features (in ONE_KB unit), along with the ranges for length of different features (exons). It will simulate the results for read coverage, and calculate predicted elongation rates for each feature. The output is a file showing the predicted/estimated elongation rates, length of each feature, etc. RIGHT NOW, WE ASSUME WE ONLY SIMULATE EXONS IN OUR SIMULATION BECAUSE WE ONLY WANT TO EVALUATE THE PERFORMANCE OF ELONGATION RATE CALCULATION.")
	parser.add_argument('--PDB', action='store_true' , help='Whether we simulate the transcription with PDB or not')
	parser.add_argument('--exon_h_list', required = True, type = float, nargs='+', help='values of the elongation rates for each exon in the gene that we are trying to simulate, in ONE_KB unit for each feature. The order of the values should be the same as the order of the features in the gtf file.')
	parser.add_argument('--length_range', nargs='+', required = True, type = float, help= 'All the possible values of length of a feature. in KB unit.')
	parser.add_argument('--default_length', required=False, type=int, default=5, help='Default length of a feature (in ONE_KB unit) that are not in the list of feat_idx_to_vary, For example, if h_list=[1,2,1], and feat_idx_to_vary=[1] then we will keep features [0,2] to be the default length throughout all simulations.')
	parser.add_argument('--feat_idx_to_vary', required=False, type=int, nargs='+', default=[0,1], help = 'indices of features for which we will vary the length, The rest of the features will have ONE default length (FOR NOW).')
	parser.add_argument('--output_fn', required=True, type=str, help = 'Where we will save the results.')
	parser.add_argument('--time', required=False, type=int, default=5, help='The interval between each labeling milestone.')
	args = parser.parse_args()
	helper.create_folder_for_file(args.output_fn)
	fail_folder = create_fail_folder(args.output_fn)
	print ('evaluate_elongation_est.py: Done getting command line argument')
	RTR = helper.DEFAULT_SIM_FEAT_LEN
	gtf_df_list = get_gtf_df_list(args.exon_h_list, args.length_range, args.feat_idx_to_vary, args.default_length, RTR)
	print('Done creating gtf_df_list! We will run {} experiments.'.format(len(gtf_df_list)))
	# apply function to run experiment and calculate the results of elongation rates for each feature
	fail_fn_list = list(map(lambda x: os.path.join(fail_folder, 'fail{}.csv'.format(x)), range(len(gtf_df_list)) ))
	result_df_list = list(map(lambda x: run_one_experiment(gtf_df_list[x], args.PDB, fail_fn_list[x]), range(len(gtf_df_list))))
	# save the results into save_fn
	result_df = pd.concat(result_df_list, ignore_index=True)
	result_df.to_csv(args.output_fn, index=False, header=True, sep=',')
	print('Done!')
