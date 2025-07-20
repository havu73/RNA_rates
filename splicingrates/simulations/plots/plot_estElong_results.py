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
	df = df[df['result_type'] == 'average']
	df['pred_h'] = df['pred_h']*1000
	df['pred_inf'] = df['pred_h'] >=10 # we took the threshold that prediction >10 implies that the elongation rate is not calculated successfully (ridiculously large)
	return df

def get_result_df(estElong_folder, filter_pred_h = True):
	'''
	tested df = get_result_df('./')
	This function read all the csv files in the estElong_folder and return a df that contains all the results
	:param estElong_folder:
	:return:
	'''
	# get all the csv files
	csv_files = glob.glob(estElong_folder + '/*/*/*.csv')
	# read all the csv files into one df
	df_list = list(map(read_one_csv, csv_files))
	df = pd.concat(df_list, ignore_index = True)
	# filter plot_df such that estimated elongation rate is calculated successfully
	if filter_pred_h:
		print('Before filtering ridiculously large pred_h, df.shape: ', df.shape)
		df = df[~df['pred_inf']]  # filter OUT cases where pred_h is ridiculously large
		print('After filtering out pred_h with extremely large values (>=10), df.shape: ', df.shape)
	df['time'] = df['length']/ONE_KB/df['true_h']
	df['gene'] = df.index // 3  # since our setting is such that we have 3 exons per gene, we will use this to identify the gene
	df = check_gene_too_long(df)
	return df

def get_all_gene_indices(row, row_idx):
	'''
	:param row: a row within the df that contains results of predictions for all cases of simulation. Here, this is a row of our interest (mostly likely bc it results in pred_h as int or ridiculously large).
	:param row_idx: the index of the row of this feature (with its true_h and pred_h) in the original df
	:return: a list of indices of the rows in the df that corresponds to the same gene as the input row, which means it includes all the features exon1,2,3 in the gene
	'''
	if row['feature'] == 'exon_1' or row['feature']=='exon1':
		return [row_idx, row_idx + 1, row_idx + 2]
	elif row['feature'] == 'exon_2' or row['feature']=='exon2':
		return [row_idx - 1, row_idx, row_idx + 1]
	elif row['feature'] == 'exon_3' or row['feature']=='exon3':
		return [row_idx - 2, row_idx - 1, row_idx]
	else:
		return []

def is_gene_too_long(gene_df, label_time = 15):
	'''
	This function takes in gene_df that typically contains 3 rows corresponding to the 3 exons of a gene. It looks at the gene and determine if the failed (ridiculously large pred_h) prediction is due the fact that some prededing exons of the failed values are just too long for the experiment to reach the gene. If so, we will return True, otherwise, False.
	:param gene_df: a df that contains the results of the 3 exons of a gene
	:param label_time: total labelling time
	:return:
	'''
	gene_df['cuml_time'] = gene_df['time'].cumsum()
	if gene_df['cuml_time'].max() < label_time:
		return False
	return True

def check_gene_too_long(weird_df, label_time = 15):
	weird_grp = weird_df.groupby('gene')
	# apply the function is_gene_too_long to each gene and return weird_df with an added column 'too_long'
	weird_df['too_long'] = weird_df['gene'].map(weird_grp.apply(is_gene_too_long))
	return weird_df
def get_weird_cases(estElong_folder, output_prefix=None):
	'''
	This function returns a list of indices of the rows in the df that have pred_h as int or ridiculously large
	:param estElong_folder:
	:return:
	'''
	df = get_result_df(estElong_folder, filter_pred_h=False)
	# now only include the rows that have pred_h as inf or ridiculously large (>=10)
	weird_df = df[df['pred_inf']]
	# get the indices of the rows that have pred_h as int or ridiculously large
	indices_to_investigate = set([])
	for idx, row in weird_df.iterrows():
		all_gene_indices = get_all_gene_indices(row, idx)
		indices_to_investigate.update(all_gene_indices)
	# sort the indices
	indices_to_investigate = sorted(list(indices_to_investigate))
	df = df.loc[indices_to_investigate]
	df['time'] = df['length']/ONE_KB/df['true_h']
	df['gene'] = df.index // 3  # since our setting is such that we have 3 exons per gene, we will use this to identify the gene
	if output_prefix != None:
		df.to_csv(output_prefix + '_weird_cases.csv', index=False)
		print(f'Saved the indices of the weird cases to {output_prefix}_weird_cases.csv')
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
	parser.add_argument('--estElong_folder', required=True, type = str, nargs='+', help='Folder where we store the results of estimated elongation rates. Structure: <estElong_folder>/*/*/*.csv')
	parser.add_argument('--length_range', nargs='+', required=False, type = float, help= 'All the possible values of length of a feature. in KB unit for which we will create a plor for.')
	parser.add_argument('--output_prefix', required=False, default=None, type=str, help='Where we will save the results.')
	args = parser.parse_args()
	helper.create_folder_for_file(args.output_prefix)
	print ('plot_estElong_results.py: Done getting command line argument')
	# first,we will get all the results input one df
	df = get_result_df(args.estElong_folder)
	# 1, plot hte scatter plot of true vs. estimated elongation rates, regladless of the length of the feature
	# second, we will plot the scatter plot of the estimated vs. ground truth elongation rates. One plot for each length tier
	# third, we will plot the results of the correlation between the estimated and ground truth elongation rates
	print('Done!')
