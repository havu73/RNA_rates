import numpy as np
import pandas as pd

def add_time_brakpoint_to_EE_junction_reads(juncReads_df, num_timepoints:int=None):
    '''
    This is a helper function that adds the time breakpoint to the junction reads dataframe.
    each junction read originally gets assigned to a time point, due to the labeling experiment.
    However, in reality, this read can be linked to two or more different time points, with the latest part of the reads
    associated with the time point that it is assigned to based on kinetic barcoding.
    This function will simply add the columns intE_at_0, intE_at_1,... time_idx to the juncReads_df such that the values are:
    - subtract_end if time_idx >= tagged_time
    - subtract_start if time_idx < tagged_time
    This function is needed simply to make the juncReads_df a particular format that can be changed later as we
    calculate the elongation rate of the intron.
    :param juncReads_df:
    :return:
    '''
    if num_timepoints is None:
        num_timepoints = juncReads_df['tagged_time'].max() + 1
    for time_idx in range(num_timepoints):
        juncReads_df[f'intE_at_{time_idx}'] = juncReads_df.apply(lambda row: row['subtract_end'] if time_idx >= row['tagged_time'] else row['subtract_start'], axis=1)
    return juncReads_df

def get_coverage_noSplice_given_juncReads_df(coverage_df, juncReads_df, num_timepoints:int=None):
    '''
    This function will take the time-tagged coverage_df and the time_tagged juncReads_df and add the junction reads to the coverage_df at regions that overlap with intron
    (these regions are associated with the subtract_start and subtract_end columns in the juncReads_df).
    The coverage_df is a dataframe that has the coverage of the reads at each time point.

    :param coverage_df: a pandas dataframe of size (g, num_timepoints) where g is the number of genomic position in the simulation
    :param juncReads_df: a pandas dataframe of size (j, 7) where j is the number of junction reads in the simulation.
    this df supposed to have the columns: 'read_index', 'feature', 'tagged_time', 'intron', 'intE_at_0', 'intE_at_1', 'intE_at_2', etc.
    It contains the information about our educated guess about whether each of the read has a certain length associated with each time point
    :return: a coverage_dataframe of size (g, num_timepoints) where the coverage of the reads ASSUMING THAT NO SPLICING OCCURS
    '''
    if num_timepoints is None:
        num_timepoints = juncReads_df['tagged_time'].max() + 1
    coverage_df_noS = coverage_df.copy()
    for idx, read in juncReads_df.iterrows():
        int_read_start = read['subtract_start']
        for time_idx in range(num_timepoints):
            coverage_df_noS.loc[int_read_start:(read[f'intE_at_{time_idx}']-1), time_idx] += 1
            int_read_start = read[f'intE_at_{time_idx}']
            if int_read_start == read['subtract_end']: # end of intron is reached, no point in checking other timepoints
                break
    return coverage_df_noS