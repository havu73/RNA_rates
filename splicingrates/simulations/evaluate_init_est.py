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
from scipy.integrate import quad
import numpy as np


# Define the integral of the transformed variable
def integral_func(y, lamb=200, k=3, lower_z=200, upper_z=300):
	'''
	X~ Weibull(lamb, k)
	Y~ unif(0,1)
	this function represents the probability that P(lower_z <= XY <= upper_z | Y=y)
	:param y:
	:param lamb:
	:param k:
	:param lower_z:
	:param upper_z:
	:return:
	'''
	lower_x = lower_z/y
	upper_x = upper_z/y
	return np.exp(-(lower_x / lamb)**k) - np.exp(-(upper_x / lamb)**k)

def prob_first_fragment_in_range(L, insertsize_min=helper.DFT_INSERTSIZE_MIN, insertsize_max=helper.DFT_INSERTSIZE_MAX, eta_val=helper.DFT_ETA_VALUE):
	'''
	:param L: the length of the gene
	:param insertsize_min:
	:param insertsize_max:
	:param eta_val:
	:return:
	'''
	k = np.log10(L) # shape of the weibull distribution
	result, error = quad(lambda y: integral_func(y, lamb=eta_val, k=k, lower_z=insertsize_min, upper_z=insertsize_max), 0, 1)
	return result, error

def get_one_output_df(pred_init, true_init, wiggleroom, lambda_init, burst_size):
	'''
	tested result_df = get_one_output_df(gtf_df_list[0],  np.array([[999, 499], [1001, 501]]), np.array([1000, 500]), 2)
	This function will create an output_df that show the results of the simulation and the predicted elongation rates.
	:param gtf_df:
	:param pred_h:
	:param nExons:
	:return:
	'''
	result_df = pd.DataFrame()
	result_df['true_init'] = true_init
	result_df['pred_init'] = list(pred_init.loc[0,:])
	result_df['wiggle_room'] = wiggleroom
	result_df['lambda_init'] = lambda_init
	result_df['burst_size'] = burst_size
	return result_df

def weibull_prob_within_range(L, insertsize_min=helper.DFT_INSERTSIZE_MIN, insertsize_max=helper.DFT_INSERTSIZE_MAX, eta_val=helper.DFT_ETA_VALUE):
	from scipy.stats import weibull_min
	scale = eta_val
	shape = np.log10(L)
	# calculate the probabilty that the fragment length is <= 300 and >= 200
	cdf_300 = weibull_min.cdf(insertsize_max, shape, scale=scale)  # P(X<=300)
	cdf_200 = weibull_min.cdf(insertsize_min, shape, scale=scale)  # P(X<=200)
	prob = cdf_300 - cdf_200
	print('prob:', prob)
	return prob


def calculate_init_from_read_coverage(coverage_df, gtf_df, startT_idx, endT_idx,insertsize_max=helper.DFT_INSERTSIZE_MAX, insertsize_min=helper.DFT_INSERTSIZE_MIN, eta_val = helper.DFT_ETA_VALUE):
	# first, get the read coverage at position 0 in each timepoint except the first one
	coverage_first_pos = coverage_df.loc[coverage_df['position'] == 0, startT_idx:endT_idx]
	# given the length of the length of the gene (gtf_df), predict the probability that a fragment will be between insrtsize_min and insertsize_max of length
	gene_length = gtf_df['end'].max() - gtf_df['start'].min()
	prob_not_within_range, _ = prob_first_fragment_in_range(gene_length, insertsize_min=insertsize_min, insertsize_max=insertsize_max, eta_val=eta_val)
	num_init_frag = coverage_first_pos / prob_not_within_range
	return num_init_frag


def run_one_experiment(burst_size, lambda_init, wiggle_room, gtf_df, PDB, fail_fn, time=5):
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
	:param lambda_init: average number of burst events per minute (burst event is the event where a burst_size transcripts are created around the same time)
	:param burst_size: number of transcripts created in a burst event
	:param wiggle_room: the wiggle room for the burst event. See the comments in function Experiment.init_bursting_transcripts to understand the meaning of this parameter. If not sure, don't modify it.
	# if I want to simulate situation such that there is no read being generated from fragments (only the whole transcripts are sequenced), the following parameters should be set carefully:
	:param simulate_cleavage:
	:param PDB:
	:param max_time_for_equilibrium:
	:param save_folder
	:return:
	'''
	label_time = np.arange(0,4) if PDB else np.arange(0,3)
	label_time = label_time * time # each label_time is <time> minute apart
	simulate_cleavage = False  ## for this problem, we don't need to simulate cleavage because we really only care about calculating the elongation speed of the transcripts. We skip splicing and cleavage for now.
	max_time_for_equilibrium = 50
	save_folder = None
	eta_val = helper.DFT_ETA_VALUE
	insertsize_min = 200
	insertsize_max = 300
	read_length = 150
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
	startT_idx = 1  # whether we have PBD or not, calculation of the initiation rate will start after the first time point
	endT_idx = len(label_time) - 1
	# try:
	pred_init_frag = calculate_init_from_read_coverage(coverage_df, gtf_df, startT_idx, endT_idx, insertsize_max=insertsize_max, insertsize_min=insertsize_min, eta_val = eta_val)  # a list of numbers: pred # new init transcripts at each timepoint, startT_idx:endT_idx
	num_transcripts = list(map(lambda x: len(x.transcripts), exp_list))
	true_init_frag = np.diff(num_transcripts, axis=0).reshape(-1)
	result_df = get_one_output_df(pred_init_frag, true_init_frag, wiggle_room, lambda_init, burst_size)
	# except:
	# 	print('Failed to calculate elongation rates for this gtf_df. We will return an empty dataframe.')
	# 	gtf_df.to_csv(fail_fn, index=False, header=True, sep=',')
	# 	result_df = pd.DataFrame(columns = ['true_init', 'pred_init', 'wiggle_room', 'lambda_init', 'burst_size'])
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
	parser = argparse.ArgumentParser(description = "This code will take in multiple values of initiation rates parameters, calculate the true and predicted initiation rates, and save the results into a csv file.")
	parser.add_argument('--PDB', action='store_true' , help='Whether we simulate the transcription with PDB or not')
	parser.add_argument('--output_fn', required=True, type=str, help = 'Where we will save the results.')
	parser.add_argument('--time', required=False, type=int, default=5, help='The interval between each labeling milestone.')
	parser.add_argument('--lambda_init', required=False, default= [2.0], type=float, nargs='+', help='The average number of burst events per minute. Multiple values are means we iterate over multiple values of lambda.')
	parser.add_argument('--burst_size', required=False, default=[10], type=int, nargs='+', help='The number of transcripts created in a burst event. Multiple values are means we iterate over multiple values of burst_size.')
	parser.add_argument('--wiggle_room', required=False, default=[0.3], type=float, nargs='+', help='The wiggle room for the burst event. Multiple values are means we iterate over multiple values of wiggle_room.')
	args = parser.parse_args()
	helper.create_folder_for_file(args.output_fn)
	fail_folder = create_fail_folder(args.output_fn)
	fail_fn = os.path.join(fail_folder, 'fail_gtf_df.csv')  # we actually should not have to use this file
	print ('evaluate_init_est.py: Done getting command line argument')
	# get the default gtf_df
	gtf_df = sim.create_variable_gtf_df(nExons=3, elong_fold_list=[1, 1, 1, 1, 1], length_fold_list=[5, 5, 5, 5, 5], SIM_FEAT_LEN=ONE_KB, RTR=helper.DEFAULT_SIM_FEAT_LEN)
	# given the different values of lambda_init, burst_size, and wiggle_room, we will iterate over all the combinations of these values
	# get the combinations of the parameters
	parameter_combinations = list(product(args.lambda_init, args.burst_size, args.wiggle_room))
	param_dict = {'lambda_init':0, 'burst_size':1, 'wiggle_room':2}
	# apply function to run experiment and calculate the results of elongation rates for each feature
	result_df_list = list(map(lambda x: run_one_experiment(x[param_dict['burst_size']], x[param_dict['lambda_init']], x[param_dict['wiggle_room']], gtf_df, args.PDB, fail_fn, time=5), parameter_combinations))
	# save the results into save_fn
	result_df = pd.concat(result_df_list, ignore_index=True)
	result_df.to_csv(args.output_fn, index=False, header=True, sep=',')
	print('Done!')
