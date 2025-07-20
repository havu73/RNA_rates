import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
from transcription.experiment import Experiment
from transcription import from_reads_to_coverage as read2cov
import matplotlib.pyplot as plt
plt.style.use('default')
plt.rcParams['figure.facecolor'] = 'white'
import helper
ONE_KB=1000
SIM_FEAT_LEN = 5000 # length of one feature in the simulation
SEED = 9999
np.random.seed(SEED)

def plot_taggedTime_junctions_lastExp(exp_list, feature='intron_1', figsize=(5,3)):
    """
    Plot the number of junction reads of the last experiment, but with the reads being stratified based on the tagged-time point from SLAM-seq
    :param exp: experiment object (most likely the last time point's experiment object)
    :param feature: what intron are we plotting?
    :param figsize: figsize of the plot (width, height)
    :return:
    """
    exp = exp_list[-1]
    endpoint_df = read2cov.get_endpoints_across_time(exp_list)  # rows: transcripts, columns: time points, values: endpoint of transcripts at that time point. Rows are ordered simply based on transcript index
    junction_df = exp.count_junction_reads(with_tagged_time=True, endpoint_df=endpoint_df)  # columns: feature, ee_reads, ie_readstime_idx
    plot_df = junction_df[junction_df['feature'] == feature]
    total_reads = plot_df[['ee_reads',  'ie_reads']].sum().sum()  # get the sum of all ee_reads and ie_reads, and then summing them to get total number of junction reads
    plot_df[['ee_reads', 'ie_reads']] = plot_df[['ee_reads', 'ie_reads']].apply(lambda x: x / total_reads)  # normalize the number of reads by the total number of reads
    fig, ax = plt.subplots(figsize=figsize)
    # Plotting ee_reads
    plt.plot(plot_df['tagged_time'], plot_df['ee_reads'], ':D', label='ee_reads', marker='o', markersize=8, color='blue')
    # Plotting ie_reads
    plt.plot(plot_df['tagged_time'], plot_df['ie_reads'], ':D', label='ie_reads', marker='o', markersize=8, color='red')
    # Adding labels and title
    ax.set_xticks(np.arange(0, len(exp_list)))  # Set x-ticks at intervals of xaxis_N
    ax.set_xticklabels(np.arange(0, len(exp_list)))  # Set x-tick labels at intervals of xaxis_N
    plt.xlabel('Time')
    plt.ylabel('Proportion of total junction reads')
    plt.title('Prop. of total time-tagged junctReads in 1 exp.: {}'.format(feature))
    plt.legend()
    plt.grid(True)
    return plot_df

def plot_junctreads_over_time(exp_list, feature='intron_1', figsize=(5,3)):
    """
    Plot the number of junction reads over time
    :param exp_list: list of experiments objects, assumption of this function is the exp inside exp_list are ordered based on time point starting from 0
    :return: a plot showing the proportion of junction reads over time
    """
    # first get the data of junction read counts
    junction_df_list = list(map(lambda x: x.count_junction_reads(with_tagged_time=False), exp_list))  # this line of code is designed such that if junction_df is already calculated it will not be calculated again, to save time
    junction_df = pd.concat(junction_df_list, axis=0)
    junction_df['ee_prop'] = junction_df['ee_reads'] / (junction_df['ee_reads'] + junction_df['ie_reads'])
    junction_df['ie_prop'] = junction_df['ie_reads'] / (junction_df['ee_reads'] + junction_df['ie_reads'])
    # now plot
    plot_df = junction_df[junction_df['feature'] == feature]
    fig, ax = plt.subplots(figsize=figsize)
    # Plotting ee_prop
    plt.plot(plot_df['time_idx'], plot_df['ee_prop'], '--r', label='ee_prop', marker='o', markersize=8, color='blue')
    # Plotting ie_prop
    plt.plot(plot_df['time_idx'], plot_df['ie_prop'], '--r', label='ie_prop', marker='o', markersize=8, color='red')
    # Adding labels and title
    ax.set_xticks(np.arange(0, len(exp_list)))  # Set x-ticks at intervals of xaxis_N
    ax.set_xticklabels(plot_df.time_idx)  # Set x-tick labels at intervals of xaxis_N
    plt.xlabel('Time')
    plt.ylabel('Proportion')
    plt.title('Proportions of junction reads over time: {}'.format(feature))
    plt.legend()
    plt.grid(True)
    return plot_df



# create a function such that I can count the number of reads overlapping each position on the genome
def _plot_barplot_for_coverage(plot_df, xaxis_N=100, title=None, xlabel: str='Position', ylabel: str = 'coverage', figsize=(6,3), cmap_color= 'PuRd', stacked=True, width:float=0.9, ax=None, grid=False, ylim=(0,300), xlim=(0,35000), save_fn=None):
    """
    Given a df showing the read coverage of each position along the genome, and stratified by categories (e.g. time points), plot the stacked bar plot
    :param plot_df: df with index: position, columns: categories, values: read coverage at each position, stratified by categories
    :param xaxis_N: the number of bars to skip between each x-axis label. We therefore only show the x-axis label for every xaxis_N bars.
    :param title: title of the plot
    :param xlabel: xlabel of the plot
    :param ylabel: ylabel of the plot
    :param cmap_color: color of the plot. Each category's bars will be a different shade of this color. Default to magenta/maroon
    :param figsize: figsize of the plot (width, height)
    :param stacked: True --> plot a stacked bar plot, False --> plot a non-stacked bar plot
    :param width: width of the bars
    :param ax: axis of the plot
    :param grid: True --> plot grid, False --> do not plot grid
    :param ylim: y-axis limits (start, end)
    :param xlim: x-axis limits (start, end)
    :return:
    """
    num_categories = len(plot_df.columns)
    # Create a color gradient
    cmap = sns.color_palette(cmap_color, n_colors=num_categories)
    colors = [cmap[i] for i in range(num_categories)]
    # plot
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    plot_df.plot(kind='bar', stacked=stacked, color=colors, ax=ax, alpha = 0.8, width=width)
    ax.set_ylim(ylim)
    print('xlim:', xlim)
    print(plot_df.tail())
    # plot_df = plot_df.loc[:6000]
    ax.set_xlim((0, 5000))  # Dont set this. Something about this will screw up the x-axis labels
    ax.set_xticks(np.arange(0, len(plot_df), xaxis_N*5))  # Set x-ticks at intervals of xaxis_N
    ax.set_xticklabels(plot_df.index[::xaxis_N*5], rotation=45)  # Set x-tick labels at intervals of xaxis_N
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(grid)
    if save_fn is not None:
        plt.savefig(save_fn)
    return

def plot_timeTagged_read_coverage(exp_list, smooth_N=1, gap_N=10, xaxis_N=100, width:float=0.8, ax=None, title='TIME-TAGGED read coverage along position for different time points', figsize=(5,3), grid=False, ylim=None, xlim=None, save_folder=None, stacked=True, time_to_plot=[], save_fn=None):
    """
    Plot the read coverage of the last time point, given that the reads can be tagged with singals showing that time point that the associated transcript was created/elongated
    :param exp_list: List of experiments. Important assumption: the experiments are ordered such that earlier experiments are earlier element sin the list of experiments. Last experiments correspond to last time point
    :param smooth_N: the number of bps to smooth over the read coverage. This is to make the plot look smoother.
    :param gap_N: the number of bps to skip between each bar. We therefore only plot the read coverage for every gap_N bps.
    :param xaxis_N: the number of bars to skip between each x-axis label. We therefore only show the x-axis label for every xaxis_N bars.
    :param time_to_plot: a list of timepoints that we would like to plot the tagged-time reads for (all the reads are from the last time point),but we did time-tagging of reads, and then we plot the read coverage that is resolved to each time point, but we will only plot time-tagged reads that are in the list of time_to_plot
    :param title: title of the plot
    :param figsize: figsize of the plot (width, height)
    :param width: width of the bars
    :param ax: axis of the plot
    :param grid: True --> plot grid, False --> do not plot grid
    :param ylim: y-axis limits (start, end)
    :param xlim: x-axis limits (start, end)
    :return: a plot showing the read coverage at the last time point in the experiment, such that the reads are time-tagged
    """
    try:
        save_fn = os.path.join(save_folder, 'timeTagged_read_coverage.csv.gz')
    except:  # save_folder is None
        save_fn = ''  # --> os.path.isfile(save_fn) is False
    if not os.path.isfile(save_fn):
        # first, get a table showing the endpoints of transcripts at each time point. Rows: transcripts, Columns: time points (in increasing order)
        endpoint_df = read2cov.get_endpoints_across_time(exp_list)  # rows: transcripts, columns: time points, values: endpoint of transcripts at that time point. Rows are ordered simply based on transcript index
        num_timepoints = len(exp_list)
        # calculate read coverage such that the counts also break down the time points at which each read was created/elongated
        coverage_df = read2cov.count_timeDep_read_coverage(exp_list[-1], endpoint_df, N=smooth_N, num_timepoints=num_timepoints)
        # coverage_df : position (along the genome), 0,1,..., num_timepoints-1 --> values are the number of reads that cover that position, broken down by the time point that the read was created/elongated
        coverage_df.set_index('position', inplace=True)  # set_index so that the df follows the default format for stacked bar plotting
        coverage_df = coverage_df.loc[::gap_N] # only plot every gap_N bps
        if save_folder != None:
            coverage_df.to_csv(save_fn, header=True, index=True, compression='gzip')
    else:
        coverage_df = pd.read_csv(save_fn, header=0, index_col=0)
    # plot
    time_to_plot = coverage_df.columns if time_to_plot==[] else time_to_plot
    plot_df = coverage_df[time_to_plot]
    if ylim is None:
        ylim = (0, plot_df.max().max())
    if xlim is None:
        xlim = (0, plot_df.index.max())
    _plot_barplot_for_coverage(plot_df, xaxis_N=xaxis_N, title=title, ylabel='time-taggged coverage', cmap_color='Blues', stacked=stacked, figsize=figsize, width=width, ax=ax, grid=grid, ylim=ylim, xlim=xlim, save_fn=save_fn)
    return coverage_df

def plot_total_read_coverage(exp_list, smooth_N:int=1, gap_N:int=50, xaxis_N:int=100, figsize=(5,3), width:float=0.8, ax=None, title='TOTAL read coverage along position for different time points', save_folder=None, grid=False, ylim=None,xlim=None, save_fn=None):
    """
    Plot the read coverage of all the experiments in exp_list
    :param exp_list: list of experiments
    :param smooth_N: the number of bps to smooth over the read coverage. This is to make the plot look smoother.
    :param gap_N: the number of bps to skip between each bar. We therefore only plot the read coverage for every gap_N bps.
    :param xaxis_N: the number of bars to skip between each x-axis label. We therefore only show the x-axis label for every xaxis_N bars.
    :param title: title of the plot
    :param figsize: figsize of the plot (width, height)
    :param width: width of the bars
    :param ax: axis of the plot
    :param grid: True --> plot grid, False --> do not plot grid
    :param ylim: y-axis limits (start, end)
    :param xlim: x-axis limits (start, end)
    :return: a plot showing the read coverage of all the experiments in exp_list
    """
    try:
        save_fn = os.path.join(save_folder, 'total_read_coverage.csv.gz')
    except:  # save_folder is None
        save_fn = ''  # --> os.path.isfile(save_fn) is False
    if not os.path.isfile(save_fn):
        coverage_df_list = list(map(lambda x: read2cov.count_total_read_coverage(x, N=smooth_N), exp_list))
        # coverage_df : position (along the genome), coverage (# reads overlapping each position), time_idx (of the experiment)
        coverage_df = pd.concat(coverage_df_list, ignore_index=True)
        coverage_df = coverage_df.pivot(index='position', columns='time_idx', values='coverage')
        coverage_df = coverage_df.loc[::gap_N] # only plot every gap_N bps
        if save_folder != None:
            coverage_df.to_csv(save_fn, header=True, index=True, compression='gzip')
    else:  # if the coverage has been saved into save_fn already
        coverage_df = pd.read_csv(save_fn, header=0, index_col=0)
    if ylim is None:
        ylim = (0, coverage_df.max().max())
    if xlim is None:
        xlim = (0, coverage_df.index.max())
    _plot_barplot_for_coverage(coverage_df, xaxis_N=xaxis_N, title=title, stacked=False, cmap_color='cubehelix', figsize=figsize, width=width, ax=ax, grid=grid, ylim=ylim, xlim=xlim, save_fn=save_fn)
    return


def plot_all_times_endsite(exp_list, ax= None, lower_idx=0, upper_idx=100000, line_width=2, y_order_by_time=False, xlim=(0,35000), save_fn=None):
    """
    Plot the end site of all transcripts at all the time points
    :param exp_list: list of experiments, each experiment contains a list of transcripts
    :param lower_idx: lower index of the transcripts to plot (we can set to plot only a subset of transcripts)
    :param upper_idx: upper index of the transcripts to plot
    :param line_width: line width of each transcript
    :param y_order_by_time: True --> the transcripts are ordered by increasing endpoints at the LAST time point first
                            False --> the transcripts are ordered by increasing endpoints at the FIRST time point first
    :param xlim: x-axis limits (start, end)
    :return: a plot showing the endpoints of all transcripts at all time points
    """
    num_timepoints = len(exp_list)
    if y_order_by_time:
        sort_by = np.arange(num_timepoints-1,-1,-1).tolist()  # [3,2,1,0]
    else:
        sort_by = np.arange(0,num_timepoints,1).tolist()  # [0,1,2,3]
    plot_df = read2cov.get_endpoints_across_time(exp_list) # rows: transcripts, columns: time_idx, value: endpoint. Rows are right now only ordered based on transcript index
    plot_df = plot_df.sort_values(by = sort_by, inplace=False, ascending=False) # if we have 4 timepoints, this will be [3,2,1,0]
    plot_df = plot_df.reset_index(drop=False)
    if ax == None:
        fig, ax = plt.subplots(figsize=(10, 6))
    sequential_colors = sns.color_palette("RdPu", 10)
    for trans_idx, row in plot_df.iterrows():
        if trans_idx >= lower_idx and trans_idx <= upper_idx:
            ax.hlines(y=trans_idx, xmin=0, xmax=row[0], colors=sequential_colors[0], lw=line_width) #darkest
            for time_idx in range(1, num_timepoints):
                linestyle = '-' if row[time_idx] > row[time_idx-1] else ':'
                ax.hlines(y=trans_idx, xmin=row[time_idx-1], xmax=row[time_idx], colors=sequential_colors[time_idx*2], lw=line_width, linestyle=linestyle) # lighter as time goes on
    plt.ylabel('transcript ordered by time: {}'.format(sort_by), fontsize=14)
    plt.xlabel('Position(bp)', fontsize=14)
    plt.xlim(xlim)
    plt.title('Lighter: transcript started at earlier time point', fontsize=16)
    plt.grid(True)
    # specify the fontsize for the title and x-axis and y-axis labels
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    if save_fn != None:
        plt.savefig(save_fn)
    return

def plot_splicing_one_timepoint(ax, exp, linewidth:int=2, max_x = 35000, max_y=1000):
    """
    Plot the splicing of transcripts at one timepoint
    :param trans_spl_df: dataframe of transcripts at one timepoint, along with information about splicing of each intron
    :return:
    - x-axis: end_site along gene
    - y-axis: transcripts
    - darker red: transcripts
    - lighter red: intron spliced out
    """
    # first get the dataframe outlining the nescessary data about each transcript in the experiment
    trans_df = exp.get_trans_df() # rows: transcripts, columns: trans_idx, time_idx, is_degrade, endpoint, intron{}_spliced for each intron
    sequential_colors = sns.color_palette("RdPu", 10)
    # given an experiment object, we would like to plot of the transcripts' end point and the splicing patterns of the introns in each of the transcript
    plot_df = trans_df.sort_values('endpoint').reset_index(drop=True)
    num_spiced_trans = [] # a list, each element correspond to an intron: number of transcripts in the exp for which this intron in spliced
    intron_start = [] # list, each element correspond to an intron: start site of the intron
    intron_end = [] # list, each element correspond to an intron: end site of the intron
    gtf_df = Experiment.gtf_df # the gtf_df used to create the experiment, shared across transcripts objects
    for intron_idx in range(exp.num_introns):
        num_spiced_trans.append(plot_df['intron{}_spliced'.format(intron_idx)].sum()) # number of transcripts in the exp for which this intron in spliced
        # note: in gtf_df, intron_idx is 1-based so we will need to add 1 to the intron_idx
        intron_start.append(gtf_df[gtf_df['feature']=='intron_{}'.format(intron_idx+1)]['start'].iloc[0])
        intron_end.append(gtf_df[gtf_df['feature']=='intron_{}'.format(intron_idx+1)]['end'].iloc[0])
    # now we plot each transcript, and whether an intron is spliced out in this transcript or not
    for trans_idx, row in plot_df.iterrows():
        # plot the whole transcript as a horizontal line
        ax.hlines(y=trans_idx, xmin=0, xmax=row['endpoint'], colors=sequential_colors[0], lw=2) #darkest
        # for each intron, if the intron is spliced out then plot a lighter dotted line on top of the intron
        for intron_idx in range(exp.num_introns):
            if row['intron{}_spliced'.format(intron_idx)]:
                ax.hlines(y=trans_idx, xmin=intron_start[intron_idx], xmax=intron_end[intron_idx], colors = sequential_colors[8], lw=linewidth, linestyle=':') # lighter
    ax.set_ylabel('Transcripts at t={}'.format(exp.time_point))
    ax.set_xlabel('End_site along gene')
    ax.set_xlim(0,max_x)
    ax.set_ylim(0,max_y)
    title = 'Timepoint:{time}, '.format(time=exp.time_point)
    title += ', '.join('i{idx}_spliced: {num_spliced}'.format(idx=x, num_spliced=num_spiced_trans[x]) for x in range(exp.num_introns))
    ax.set_title(title)
    ax.grid(True)
    return ax

def plot_splicing_multiple_timepoints(exp_list, nrows:int=4, ncols:int=1, max_x=35000, max_y = 1000, linewidth:int=2):
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5*ncols, 4*nrows))
    for ax_idx, ax_new in enumerate(axs.flatten()):
        plot_splicing_one_timepoint(ax_new, exp_list[ax_idx], linewidth=linewidth, max_x=max_x, max_y=max_y)
    fig.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.4, hspace=0.4)
    return fig

def plot_elongation_two_timepoints(prev_exp, curr_exp, stop_site_range=(0, 500), sort_values = [1,0], xmax=35000):
    # filter all transcripts that stop before stop_site_t0 at first time point
    prev_df = prev_exp.get_trans_df()
    curr_df = curr_exp.get_trans_df()
    prev_df = prev_df[(prev_df['endpoint'] >= stop_site_range[0]) & (prev_df['endpoint'] <= stop_site_range[1])]
    # filter all transcripts that are the same transcripts as those filtered at t0
    curr_df = curr_df[curr_df['trans_idx'].isin(prev_df.trans_idx)]
    df = pd.concat([curr_df, prev_df])
    plot_df = df.pivot(index='trans_idx', columns='time_idx', values='endpoint')
    plot_df = plot_df.astype(int) #turn the float values into integers
    plot_df = plot_df.sort_values(by = sort_values)
    plot_df = plot_df.reset_index()
    plt.figure(figsize=(10, 6))
    sequential_colors = sns.color_palette("RdPu", 10)
    for trans_idx, row in plot_df.iterrows():
        linestyle = '-' if row[sort_values[1]] > row[sort_values[0]] else ':' # the endpoint is only smaller than from the previous time point if the transcript is cleaved
        plt.hlines(y=trans_idx, xmin=0, xmax=row[sort_values[0]], colors=sequential_colors[0], lw=4) #darkest
        plt.hlines(y=trans_idx, xmin=row[sort_values[0]], xmax=row[sort_values[1]], colors=sequential_colors[8], lw=4, linestyles=linestyle) # medium
    plt.xlim(0,xmax)
    plt.ylabel('transcript index')
    plt.xlabel('End_site along gene')
    plt.title('Darker: transcript started at earlier time point, lighter: elongated after 5 mins')
    plt.grid(True)
    plt.show()

def calculate_cleavage_stats(exp_list):
    """
    Calculate the number of cleaved and uncleaved transcripts at each time point
    :param exp_list: list of experiments objects, assumption of this function is the exp inside exp_list are ordered based on time point starting from 0
    :return: a dataframe with columns: time_idx,
    """
    num_timepoints = len(exp_list)
    # first find the range of the coordinates that overlap with the PAS and the RTR period
    gtf_df = Experiment.gtf_df
    PAS_end = gtf_df[gtf_df['feature']=='PAS']['end'].iloc[0]
    df_list = list(map(lambda x: x.get_trans_df(), exp_list))  #columns: trans_idx, time_idx, is_degrade, endpoint, is_cleaved
    endpoint_df = pd.concat(df_list).pivot(columns='time_idx', index='trans_idx', values='endpoint').fillna(0)
    cleaved_df = pd.concat(df_list).pivot(columns='time_idx', index='trans_idx', values='is_cleaved').fillna(False)
    result_df = pd.DataFrame(columns=['pastPAS_uncleaved', 'num_total_cleaved', 'newly_cleaved', 'cleaved_from_prev_pastPAS'])
    for exp_idx, df in enumerate(df_list):
        pastPAS_uncleaved = ((~df['is_cleaved']) & (df['endpoint'] >= PAS_end)).sum()
        num_total_cleaved = df['is_cleaved'].sum()
        if exp_idx == 0:
            newly_cleaved = 0
            cleaved_from_prev_pastPAS = 0
        else:
            cleaved_from_prev_pastPAS = ((~cleaved_df[exp_idx-1]) & (cleaved_df[exp_idx]) & (endpoint_df[exp_idx-1] >= PAS_end)).sum()
            newly_cleaved = (cleaved_df[exp_idx] & (~cleaved_df[exp_idx-1])).sum()
        result_df.loc[exp_idx] = pd.Series({'pastPAS_uncleaved': pastPAS_uncleaved, 'num_total_cleaved': num_total_cleaved, 'newly_cleaved': newly_cleaved, 'cleaved_from_prev_pastPAS': cleaved_from_prev_pastPAS})
    result_df['time_idx'] = result_df.index
    result_df['prop_cleaved_from_prev_pastPASuncleaved'] = result_df['cleaved_from_prev_pastPAS'] / result_df['pastPAS_uncleaved'].shift(1)
    result_df['prop_cleaved_from_prev_pastPASuncleaved'].fillna(0, inplace=True)
    return result_df