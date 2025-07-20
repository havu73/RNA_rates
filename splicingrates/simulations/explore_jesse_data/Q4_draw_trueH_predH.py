import os
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import argparse
plt.style.use('default')
plt.rcParams['figure.facecolor'] = 'white'
ONE_KB=1000
SIM_FEAT_LEN = 5000 # length of one feature in the simulation
SEED = 9999
np.random.seed(SEED)


def draw_one_binH_idx(pred_df, index, gene_name, data_folder):
    binH_dict = {0: 0.1, 1: 0.5, 2: 1, 3: 2}  # index, bin of elongation rate in kb
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    # get only data with index == 0
    plot_df = pred_df[pred_df['index'] == index].copy()
    # now, draw the line plot, each line is a different methods (true_h, simpleSmooth, etc.), x axis: start, y axis: h
    ax.plot(plot_df['start'], plot_df['true_h'], label='true_h', color='green', linestyle='--', alpha=1)
    # color map for different methods
    color_map = sns.color_palette("bright", 5)
    ax.scatter(plot_df['start'], plot_df['simpleSmooth'], label='simpleSmooth', color=color_map[0], alpha = 0.2, s=5)
    ax.scatter(plot_df['start'], plot_df['simpleSolver'], label='simpleSolver', color=color_map[1], alpha = 0.2, s=5)
    ax.scatter(plot_df['start'], plot_df['bayesLinearSolver'], label='bayesLinearSolver', color=color_map[2], alpha = 0.2, s=5)
    ax.scatter(plot_df['start'], plot_df['bayesRBFSolver'], label='bayesRBFSolver', color=color_map[3], alpha = 0.2, s=5)
    ax.scatter(plot_df['start'], plot_df['logNormalSolver'], label='logNormalSolver', color=color_map[4], alpha = 0.2, s=5)
    ax.set_xlabel('start')
    ax.set_ylabel('h')
    ax.set_title(f'{gene_name} - binH: {binH_dict[index]}')
    ax.legend()
    plt.savefig(f'{data_folder}/{gene_name}/pred_h_{index}.png')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Estimate elongation rates")
    parser.add_argument('--gene_name', type=str, required=True, help="Gene name")
    parser.add_argument('--data_folder', type=str, required=True, help="Folder containing the data")
    args = parser.parse_args()
    pred_fn = f'{args.data_folder}/{args.gene_name}/pred_h.csv.gz'
    pred_df = pd.read_csv(pred_fn, header = 0, index_col = None, sep = '\t')
    h_breaks = [100, 500, 1000, 2000]
    for i in range(4):
        draw_one_binH_idx(pred_df, i, args.gene_name, args.data_folder)

