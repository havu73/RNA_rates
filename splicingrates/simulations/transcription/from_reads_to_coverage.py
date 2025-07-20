import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('default')
plt.rcParams['figure.facecolor'] = 'white'
import helper
ONE_KB=1000
SIM_FEAT_LEN = 5000 # length of one feature in the simulation
SEED = 9999
np.random.seed(SEED)

def get_endpoints_across_time(exp_list):
    """
    This is a helper function that will return a dataframe of the endpoints of each transcript across multiple time points (captured in exp_list)
    :param exp_list: list of experiments objects, assumed to be ordered such that the first exp object corresponds to the first time point
    :return: df showing endpoint of transcripts across multiple time points. Rows: transcritps, columns: time_idx, value: endpoint
    """
    df = pd.concat(list(map(lambda x: x.get_trans_df(), exp_list)))  #columns: trans_idx, time_idx, is_degrade, endpoint
    # this exp.trans_df will include all transcripts, including those that are mature and degraded.
    # at the last time point, there can be transcripts with endpoint=0. That is because the transcript was created at time very close to the end of the experiment (wiggle_room contributes to this), and therefore the transcript did not have time to elongate
    # we will remove these transcripts from the dataframe
    df = df[df['endpoint'] > 0]
    # Pivot the table
    pivot_df = df.pivot(index='trans_idx', columns='time_idx', values='endpoint').fillna(0)
    # if the transcript is not degraded at any time point, then it is not degraded
    is_degrade_any = df.groupby('trans_idx')['is_degrade'].any()
    pivot_df = pivot_df.join(is_degrade_any)  # columns: 0,1,2, .. <time_idx>, is_degrade
    pivot_df = pivot_df[~pivot_df['is_degrade']].drop('is_degrade', axis=1)  # only plot transcripts that are not degraded
    pivot_df = pivot_df.astype(int) #turn the float values into integers
    # if the column 0 is not present, then add it with all values being 0.
    # it happens when all our transcripts were created at time_idx > 0
    if 0 not in pivot_df.columns:
        pivot_df[0] = 0
    return pivot_df

def create_coverage_df_from_reads(exp, num_timepoints):
    gene_start = exp.gtf_df['start'].iloc[0]
    gene_end = exp.gtf_df['end'].iloc[-1]
    reads_end = exp.reads_df['abs_end'].max() # the end of the last read in this experiment
    last_index = max(gene_end, reads_end)  # the end of the last read or the end of the gene, whichever is later
    coverage_df = pd.DataFrame(index = np.arange(gene_start, last_index), columns=np.arange(num_timepoints), data=0)
    # df = exp.reads_df[exp.reads_df['abs_start'] == 0] # only consider reads that are tagged with a time point that is less than the number of time points that were profiled in SLAM-seq experiment
    for idx, row in exp.reads_df.iterrows():  # dont change anything about this for loop even if it looks bulky, it has been tested and debugged
        if row['abs_start'] > last_index:
            continue
        if row['abs_end'] > last_index:
            coverage_df.loc[row['abs_start']:(row['abs_end']-1), row['tagged_time']] += 1
            if row['subtract_start'] < row['subtract_end']:
                coverage_df.loc[row['subtract_start']:(row['subtract_end']-1), row['tagged_time']] -= 1
            continue
        try:
            coverage_df.loc[row['abs_start']:(row['abs_end']-1), row['tagged_time']] += 1
        except:
            print(row)
            import sys
            sys.exit(1)
        if row['subtract_start'] < row['subtract_end']:
            coverage_df.loc[row['subtract_start']:(row['subtract_end']-1), row['tagged_time']] -= 1
    coverage_df = coverage_df.fillna(0)
    coverage_df['position'] = coverage_df.index
    coverage_df = coverage_df.astype(float)
    return coverage_df


def count_timeDep_read_coverage(exp, endpoint_df, num_timepoints=4):
    """
    This function will count the number of reads that overlap each bp along the gene in the experiment, given that the reads maybe have been produced based on SLAM-seq procedure so each read can be time-resolved (tagged to a time point as its creation time)
    :param exp: experiment object representing this experiment. We can call on exp.get_reads_df() to generate the reads_df if it is empty
    Then, we can call on exp.tag_reads_by_timepoint() to tag each read with the time point that it was created at
    :param endpoint_df: dataframe of endpoints of transcripts across multiple time points. Rows: transcritps, columns: time_idx, value: endpoint
    :param N: the number of bps to smooth over the read coverage. This is to make the plot look nicer.
    :param num_timepoints: the number of time points that were profiled in SLAM-seq experiment
    :return: coverage with columns: position, 0, 1, ... num_timepoints-1 --> read coverage at each position, tagged at each time point
    """
    #### given reads_df: abs_start, abs_end, subtract_start, subtract_end, tagged_time
    #### calculate coverage_df: position, 0,1,..., num_timepoints-1 --> values are the number of reads that cover that position, broken down by the time point that the read was created/elongated
    #### tested: assert that read coverage at position 0 is similar to number of tagged reads at exp_list[-1]
    # assert that the columns of endpoint_df are in increasing order --> this assertion does not work all the time beacuse there are cases where I do not want to plot the first time point (such as PDB experiments)
    # assert np.all(set(endpoint_df.columns) == set(np.arange(num_timepoints)))
    # obtain the reads from the transcripts that are present in the experiment
    exp.get_reads_df() # this will only take time if exp.reads_df is empty, then this will generate the reads_df to obtain the reads generated from this experiment. Otherwise, do nothing
    # for each read, determine the time point that the transcript was created/elongated, aka tagged in the SLAM-seq experiment (see comment in the function for more details)
    exp.tag_reads_by_timepoint(endpoint_df) # tag each read with the time point that it was created at. column 'tagged_time' is added to exp.reads_df
    coverage_df = create_coverage_df_from_reads(exp, num_timepoints)
    return coverage_df

# create a function such that I can count the number of reads overlapping each position on the genome
def count_total_read_coverage(exp, N = 10):
    """
    This function will count the number of reads that overlap each bp along the gene
    This function is different from function count_timeDep_read_coverage in that this function will count the total number of reads overlapping each bp along the gene, regardless of the time point that each read is tagged with
    :param exp: experiment object representing this experiment. We can call on exp.get_reads_df() to generate the reads_df if it is empty
    :param N: the number of bps to smooth over the read coverage. This is to make the plot look nicer.
    :return: coverage with columns: position, coverage, time_idx
    """
    exp.get_reads_df() # if exp.reads_df is empty, then this will generate the reads_df to obtain the reads generated from this experiment
    gene_start = exp.gtf_df['start'].iloc[0]
    gene_end = exp.gtf_df['end'].iloc[-1]
    coverage = pd.Series(index = np.arange(gene_start, gene_end+1), data=0)
    for idx, row in exp.reads_df.iterrows():  # dont change anything about this for loop even if it looks bulky, it has been tested and debugged
        if row['abs_start'] >= gene_end:
            continue
        if row['abs_end'] > gene_end:
            coverage[row['abs_start']:(gene_end-1)] += 1
            if row['subtract_start'] < row['subtract_end']:
                coverage[row['subtract_start']:(row['subtract_end']-1)] -= 1
            continue
        coverage[row['abs_start']:(row['abs_end']-1)] += 1
        if row['subtract_start'] < row['subtract_end']:
            coverage[row['subtract_start']:(row['subtract_end']-1)] -= 1  # spliced out region in this read (it is a exon-exon junction)
    if N>1:
        coverage = coverage.rolling(window=N, center=True).mean()
    coverage = coverage.fillna(0)
    coverage = coverage.astype(float)
    coverage = coverage.to_frame(name='coverage')
    coverage['time_idx'] = exp.time_point
    coverage['position'] = coverage.index
    return coverage

def correct_first_fragment_effects(exp, coverage_df):
    """
    This function will correct for the edge effect of the first fragment
    :param exp: experiment object representing this experiment. We can call on exp.get_reads_df() to generate the reads_df if it is empty
    :param coverage_df: dataframe of read coverage across the gene. columns: position, 0,1,..., num_timepoints-1 --> read coverage at each position, tagged at each time point
    :return: corrected coverage_df
    """
    exp.get_reads_df() # if exp.reads_df is empty, then this will generate the reads_df to obtain the reads generated from this experiment
    max_transcript_len = exp.reads_df['abs_end'].max()
    reads_df_to_consider = exp.reads_df if not exp.pair_end else exp.reads_df[exp.reads_df['pairedEnd_type'] == 'first']
    reads_df_to_consider = reads_df_to_consider[reads_df_to_consider['abs_start'] <= max_transcript_len]
    # we will only look at transcripts that has abs_start within the 75th percentile of possible read length
    from scipy.stats import weibull_min
    scale = exp.eta_val
    shape = np.log10(max_transcript_len)
    threshold_start = weibull_min.ppf(0.75, c=shape, scale=scale)
    edgeCorr_cov_df = coverage_df.copy()
    for idx, row in reads_df_to_consider.iterrows():
        # assuming that the first fragment is right before this read, we will calculate the probability that the probability that the uniform break of the first fragment will result in one of the smaller fragments pass the threshold
        if row['abs_start'] < threshold_start:
            continue
        elif exp.insertsize_min <= row['abs_start'] <= exp.insertsize_max:
            # the first fragment should have passed the length filter
            # but bc of the edge effect, it may not have passed the length filter.
            # if p is the probability that a part of this fragment passed the length filter, then we wil add 1-p to each position leading up to this position
            # import pdb; pdb.set_trace()
            p = (row['abs_start']-exp.insertsize_min)/row['abs_start']
            edgeCorr_cov_df.loc[:row['abs_start'], row['tagged_time']] += 1-p
        else: # row['abs_start'] > exp.insertsize_max
            continue # ignore for now, even through the right thing to do may have been to account for the edge effect of the first fragment
    return edgeCorr_cov_df