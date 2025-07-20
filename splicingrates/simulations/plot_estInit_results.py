import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import argparse
import helper
ONE_KB = 1000
length_tiers = [[0.5, 1, 1.5],
                [2, 3.5 ,5],
                [6,9,12],
                [15,22.5, 30]] # in kb

def read_one_csv(csv_file):
	df = pd.read_csv(csv_file, header=0, index_col=None, sep=',')
	return df

def get_result_df(estInit_folder):
	'''
	tested df = get_result_df('./')
	This function read all the csv files in the estInit_folder and return a df that contains all the results
	:param estInit_folder:
	:return:
	'''
	# get all the csv files
	print(estInit_folder)
	csv_files = glob.glob(estInit_folder + '/*.csv')
	# read all the csv files into one df
	df_list = list(map(read_one_csv, csv_files))
	df = pd.concat(df_list, ignore_index = True)
	# filter plot_df such that estimated elongation rate is calculated successfully
	return df




def plot_all_elongation(df, output_prefix= None):
	'''
	This function plot the scatter plot of the estimated vs. ground truth elongation rates, regardless of the length of the feature
	:param df:
	:return:
	'''
	fig, ax = plt.subplots()
	print('After filtering out pred_h with extremely large values (>=10), df.shape: ', df.shape)
	sns.boxplot(df, x='true_h', y = 'pred_h', ax=ax)
	id_x = np.sort(df['true_h'].unique())
	numeric_mapping = {val: idx for idx, val in enumerate(id_x)}
	x_positions = [numeric_mapping[val] for val in id_x]
	id_y = id_x
	plt.plot(x_positions, id_y, color = 'red', marker = 'o', linestyle = 'None')
	ax.set_xlabel('True elongation rate')
	ax.set_ylabel('Estimated elongation rate')
	ax.set_title('Scatter plot of true vs. estimated elongation rates')
	if output_prefix != None:
		plt.savefig(output_prefix + '_all.png')
	else:
		print('No output_prefix is provided. The plot will not be saved.')
	return

def plot_length_stratified(df, output_prefix=None):
	# for each length of the feature, plot the scatter plot of true_h vs. pred_h
	axes, fig = plt.subplots(len(length_tiers), 3, figsize=(15, 15))
	for i in range(len(length_tiers)):
		for j, length in enumerate(length_tiers[i]):
			length = length * ONE_KB
			plot_df = df[df['length'] == length]
			ax = fig[i][j]
			sns.boxplot(plot_df, x='true_h', y='pred_h', ax=ax)
			id_x = np.sort(plot_df['true_h'].unique())
			numeric_mapping = {val: idx for idx, val in enumerate(id_x)}
			x_positions = [numeric_mapping[val] for val in id_x]
			id_y = id_x
			# plot just the dots of the TRUE values x and y
			ax.plot(x_positions, id_y, color='red', marker='o', linestyle='None')
			ax.set_title('Length = ' + str(length) + ' bp')
			ax.set(xticks=x_positions, xticklabels=id_x, ylim=(0, 8))
	plt.tight_layout()
	if output_prefix != None:
		plt.savefig(output_prefix + '_length_stratified.png')
	else:
		print('No output_prefix is provided. The plot will not be saved.')
	return

def length_Udist_stratified(df, output_prefix=None, length_tier_idx=0):
	length_to_plot = length_tiers[length_tier_idx]
	length_to_plot = ONE_KB * np.array(length_to_plot)
	plot_df = df[df['length'].isin(length_to_plot)]
	num_rows = len(plot_df['Udist'].unique())
	num_cols = len(length_to_plot)
	axes, fig = plt.subplots(num_rows, num_cols, figsize=(5 * num_cols, 5 * num_rows))
	for i, Udist in enumerate(plot_df['Udist'].unique()):
		for j, length in enumerate(length_to_plot):
			ax = fig[i, j]
			plot_df = (df[(df['Udist'] == Udist) & (df['length'] == length)]).copy() # to shut off the warning
			if plot_df.shape[0] == 0:
				print(f'No data for Udist={Udist}, length={length}')
				continue
			# draw the scatter plot
			id_x = np.sort(plot_df['true_h'].unique())
			numeric_mapping = {val: idx for idx, val in enumerate(id_x)}
			x_positions = [numeric_mapping[val] for val in id_x]
			id_y = id_x
			# create the x and y positions such that x are the indices of the unique values of true_h, while y are the unique values of pred_h for the corresponding true_h in plot_df
			plot_df['x_pos'] = plot_df['true_h'].map(numeric_mapping)
			ax.plot(plot_df['x_pos'], plot_df['pred_h'], color='blue', marker='o', linestyle='None', alpha=0.3, markersize=15)
			# plot just the dots of the TRUE values x and y
			ax.plot(x_positions, id_y, color='red', marker='o', linestyle='None', markersize=15, alpha=0.3)
			ax.set_title(f'Udist={Udist}, length={length}')
			ax.set(xticks=x_positions, xticklabels=id_x, ylim=(0, 8))
	plt.tight_layout()
	if output_prefix != None:
		plt.savefig(output_prefix + f'_length_Udist_stratified{length_tier_idx}.png')
	else:
		print('No output_prefix is provided. The plot will not be saved.')
	return

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description = "This code take in the results of estimated vs. ground truth elongation rates and plot the results.")
	parser.add_argument('--PDB', action='store_true', help='Whether we simulate the transcription with PDB or not')
	parser.add_argument('--estInit_folder', required=True, type = str, help='Folder where we store the results of estimated elongation rates. Structure: <estInit_folder>/*/*/*.csv')
	parser.add_argument('--output_prefix', required=False, default=None, type=str, help='Where we will save the results.')
	args = parser.parse_args()
	helper.create_folder_for_file(args.output_prefix)
	print ('plot_estElong_results.py: Done getting command line argument')
	# first,we will get all the results input one df
	df = get_result_df(args.estInit_folder)
	# 1, plot hte scatter plot of true vs. estimated elongation rates, regladless of the length of the feature
	# second, we will plot the scatter plot of the estimated vs. ground truth elongation rates. One plot for each length tier
	# third, we will plot the results of the correlation between the estimated and ground truth elongation rates
	print('Done!')
