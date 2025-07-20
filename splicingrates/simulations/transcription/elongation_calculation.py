import numpy as np
import pandas as pd
import helper

def get_endpoints_from_gtf(elongf_df, convert_to_KB=False):
    '''
    This function will create a list of endpoints of features from elongf_df. The output is needed to calculate the coefficient matrix A
    :param elongf_df: df of the gene annotation file
    :param convert_to_KB: True if we want to convert genomic position to KB. This will help the calculation of h to be more numerically stable. Output will be in KB/min
    :return: endpoints
    '''
    # get the endpoints of features in the elongf_df
    # get the endpoints of features in the elongf_df
    endpoints = elongf_df[['start', 'end']].values
    # assert that the end of row i is the same as the start of row i+1
    assert np.all(endpoints[:-1, 1] == endpoints[1:, 0]), 'the endpoints of features in elongf_df should be continuous. Here, it is not, please check the elongf_df'
    endpoints = endpoints[:,0] #np.concatenate((endpoints[:, 0], endpoints[-1, 1]))  # the last endpoint of the last feature is the end of the gene
    # add np.inf to the end if it is not already there
    if endpoints[-1] != np.inf:
        endpoints = np.append(endpoints, np.inf) # add a very large number to the end of the endpoints list to make sure that the last segment (beyond the end of all features) is included in the calculation
    if convert_to_KB:
        endpoints = endpoints/1000
    return endpoints

def find_feature_overlap(x0, x1, feat_endpoints):
    '''
    Given the gene coordinates x0 and x1, find the length of overlap between [x0,x1] and the features
    :param prev_stop:
    :param curr_stop:
    :param feat_endpoints:
    :return:
    '''
    this_x0 = x0
    A = np.array([0]* (len(feat_endpoints)-1))  # A is the vector showing the length of overlap of [x0-x1] with each feature
    for i in range(len(feat_endpoints) - 1):
        if this_x0 < feat_endpoints[i]: # if x0 starts out < gene_start
            # if otherwise, this_x0 will be more likely >= feat_endpoints[i]
            break
        if this_x0 > feat_endpoints[i + 1]:  # this entry starts after the end of this feature
            continue
        if x1 < feat_endpoints[i]:  # this entry ends before the start of this feature
            break  # no need to continue since A is initally filled with zeros
        if this_x0 >= feat_endpoints[i] and this_x0 < feat_endpoints[i + 1]:  # this entry starts within this feature
            if x1 > feat_endpoints[i + 1]:  # this entry ends after the end of this feature
                A[i] = feat_endpoints[i + 1] - this_x0
                this_x0 = feat_endpoints[i + 1]
                continue  # go to the next feature
            else:  # this entry ends within this feature
                A[i] = x1 - this_x0
                break  # no need to continue to the following features since A is initally filled with zeros
    return A

def time_to_elongate(prev_stop, curr_stop, elongf_df, e_colname='txrate'):
    '''
    Given the current and previous stop site of transcripts along the gene, calculate the time it takes for the
    transcript to elongate from prev_stop to curr_stop given the ground-truth/estimated elongation rate
    :param prev_stop:
    :param curr_stop:
    :param elongf_df:
    :param e_colname:
    :return:
    '''
    if prev_stop == curr_stop:
        return 0
    if prev_stop > curr_stop:  # this can happen when I call the functions to calculate splicing probability, and the prev_stop is < intronEnd beacuse the transcript has not reached end-of-intron yet.
        # instead of raising an error, I will just return 0 so that unspliced probability is absolutely 1
        return 0
    if prev_stop < elongf_df['start'].min():
        raise ValueError('prev_stop is smaller than the start of the gene')
    if prev_stop >= elongf_df['end'].max(): # if prev_stop is beyond the end of the gene
        txrate = elongf_df.iloc[-1][e_colname]  # if beyond the end of gene, assign the txrate of the last feature of the gene
        return (prev_stop - curr_stop) / (txrate * helper.ONE_KB)
    else: # prev_step within the gene
        feat_endpoints = get_endpoints_from_gtf(elongf_df, convert_to_KB=False) # [gene_start (0), end_of_feat1, end_of_feat2, ..., end_of_PAS, np.inf] # the np.inf includes the RTR
        overlap_feat = find_feature_overlap(prev_stop, curr_stop, feat_endpoints) # [overlap exon1, ...., overlap_RTR] --> length exactly the same as the elongf_df
        if len(overlap_feat) != len(elongf_df):
            raise ValueError('The length of overlap_feat is not the same as the length of elongf_df')
        return np.sum(overlap_feat/elongf_df[e_colname]/helper.ONE_KB)


def calculate_startsite_given_pred_h(curr_stop, elongf_df, e_colname='pred_e', time_since_prev:int=5):
    '''
    Given the current stop size, calculate the potential stop size of the transcript in previous time point,
    given predicted elongation rate, the time, and the gene annotation elongf_df
    :param curr_stop: coordinate of the current stop site
    :param elongf_df: gene annotation df. columns: chromosome, source, feature, start, end, length, txrate (in kb per minute), time, sequence, is_intron, intron_h
    :param e_colname: the column name in elongf_df that contains the predicted elongation rate of features
    :param time_since_prev: minutes since the previous time point
    :return:
    '''
    if time_since_prev == 0:
        return curr_stop
    curr_feat_idx = elongf_df[(elongf_df['start'] <= curr_stop) & (elongf_df['end'] > curr_stop)].index  # the index of the feature that the current stop site is in. It should be either 0 or 1 feature that the stop site is in.
    assert len(curr_feat_idx) <= 1, 'There are >1 features in elongf_df that overlap with each other around position {}'.format(curr_stop)
    if len(curr_feat_idx) == 0:  # this means that the previous time point stop site is outside of the last feature of the gene. We assume that it will keep elongate with the same rate as the last feature
        curr_feat_idx = elongf_df.index[-1]  # the index of the last feature of the gene. type <class 'pandas.core.indexes.range.RangeIndex'> can be accessed using index [-1]
    else:  # if there is only one feature that the current time point stop site is in (len(prev_feat_idx) == 1)
        curr_feat_idx = curr_feat_idx[0]  # the index of the feature that the current time point stop site is in
    elong_time = time_since_prev
    curr_stop = curr_stop
    while elong_time > 0:
        try:
            time_till_end_feat = (curr_stop - elongf_df.loc[curr_feat_idx, 'start']) / (elongf_df.loc[curr_feat_idx, e_colname] * helper.ONE_KB)  # time till the end of the current feature
        except: # the only time that the above code would fail if curr_feat_idx == -1 which means we reached the beginning of the gene to return the coordinate of the gene start
            return elongf_df.iloc[0,'start']
        if elong_time >= time_till_end_feat:  # if the elongation time is longer than the time till the start of the current feature
            elong_time = elong_time - time_till_end_feat  # the remaining elongation time
            curr_stop = elongf_df.loc[curr_feat_idx, 'start']  # the start of the current feature
            curr_feat_idx = curr_feat_idx - 1  # move to the previous feature
        else:  # if the elongation time is shorter than the time till the end of the current feature
            curr_stop = curr_stop - elong_time * elongf_df.loc[curr_feat_idx, 'txrate'] * helper.ONE_KB
            elong_time = 0
    return curr_stop


def calculate_enlongated_endsite(prev_stop, elongf_df, time_since_prev):
    """
    This function will calculate the stop site of A transcript NOW, given the time has passed since the previous time point and the stop site of the transcript at the previous time point (prev_stop), and the transcription rate of features (intron, exon,etc.) along the gene
    :param prev_stop: the ABSOLUTE coord of stop site of the transcript at the previous time point
    :param elongf_df: gene annotation df. columns: chromosome, source, feature, start, end, length, txrate (in kb per minute), time, sequence, is_intron, intron_h
    :param time_since_prev: minutes since the previous time point (with prev_stop) until NOW. Based on this, we want to calculate the stop site of the transcript at this time point
    :return: the stop site of the transcript at this time point
    """
    if time_since_prev==0:
        return prev_stop
    # first find the features that the previous time point stop site is in
    prev_feat_idx = elongf_df[(elongf_df['start']<=prev_stop) & (elongf_df['end']>prev_stop)].index  # the index of the feature that the previous time point stop site is in. It should be either 0 or 1 feature that the stop site is in.
    assert len(prev_feat_idx) <= 1, 'There are >1 features in elongf_df that overlap with each other around position {}'.format(prev_stop)
    if len(prev_feat_idx) == 0:  # this means that the previous time point stop site is outside of the last feature of the gene. We assume that it will keep elongate with the same rate as the last feature
        prev_feat_idx = elongf_df.index[-1]  # the index of the last feature of the gene. type <class 'pandas.core.indexes.range.RangeIndex'> can be accessed using index [-1]
    else:  # if there is only one feature that the previous time point stop site is in (len(prev_feat_idx) == 1)
        prev_feat_idx = prev_feat_idx[0]  # the index of the feature that the previous time point stop site is in
    # calculate the current stop site
    # assumptions: (1) transcript can elongate beyond the last feature of the gene (imaginary)
    # (2) transcription rate beyond the end of the last feature is similar to the transcription rate of the last feature
    # this means, we have not done anything to simulate the cleavage event yet
    if prev_stop >= elongf_df.end.iloc[-1]:  # if the previous stop is already beyond of the end of the gene
        # it means prev_feat_idx is the last feature of the gene, so assert that
        assert prev_feat_idx == elongf_df.index[-1], 'prev_stop is beyond the end of the gene, but prev_feat_idx is not the last feature of the gene. Function failed: calculate_stopsite_in_elongation'
        curr_stop = prev_stop + time_since_prev * elongf_df.loc[prev_feat_idx, 'txrate'] * helper.ONE_KB # keep the txrate of the last feature
        # even if the last feature in elongf_df is a RTR, this curr_stop is still considered legal because we assume that the transcript can elongate beyond the end of the gene
    else:  # if the previous stop is within the gene
        elong_time = time_since_prev
        curr_stop = prev_stop
        while elong_time > 0:
            try:
                time_till_end_feat = (elongf_df.loc[prev_feat_idx, 'end'] - curr_stop) / (elongf_df.loc[prev_feat_idx, 'txrate'] * helper.ONE_KB)  # time till the end of the current feature
            except: # the only time that the above code would fail if prev_feat_idx == elongf_df.shape[0] which means we reached the end of the gene
                # calculate the remaining elongation time, given the rate of the last feature in the gene
                prev_feat_idx = prev_feat_idx - 1  # move to the last feature of the gene
                curr_stop = curr_stop + elong_time * elongf_df.loc[prev_feat_idx, 'txrate'] * helper.ONE_KB
                elong_time = 0
                break  # get out of the while loop
            if elong_time >= time_till_end_feat:  # if the elongation time is longer than the time till the end of the current feature
                elong_time = elong_time - time_till_end_feat  # the remaining elongation time
                curr_stop = elongf_df.loc[prev_feat_idx, 'end']  # the end of the current feature
                prev_feat_idx = prev_feat_idx + 1  # move to the next feature
            else:  # if the elongation time is shorter than the time till the end of the current feature
                curr_stop = curr_stop + elong_time * elongf_df.loc[prev_feat_idx, 'txrate'] * helper.ONE_KB
                elong_time = 0
    return curr_stop

