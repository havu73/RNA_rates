import pandas as pd
import numpy as np
import argparse
import helper
from transcripts import Transcript, calculate_enlongated_endsite
from scipy.special import gamma

DEFAULT_SEED = 9999
DFT_LAMBDA_INIT = 0.5
DFT_BURST_SIZE = 5
DFT_WIGGLE_ROOM = 0.1
def set_seed(seed=DEFAULT_SEED):
    """
    Set the seed for the random number generator
    :param seed: seed for the random number generator
    :return: None
    """
    np.random.seed(seed)

def read_gtf_file(gtf_fn, intron_h, PAS_h, RTR):
    """
    read in the gtf file and do some sanity check of the input
    Assumption: this file will write features such that the start coordinate of the first feature is the start of the gene, and coordinate is all 1-based, and the features are written [start,end]. But this function will change such that the start and end coodinate is 0-based, and each feature coordinate is such that [start, end).
    :param gtf_fn: gene annotation file. chromosome, source, feature, start, end, txrate, time, sequence
    :param intron_h: default intron half life, in case the gtf file does not have this information
    :param PAS_h: default cleavage half life, in case the gtf file does not have this information
    :param RTR: default readthrough region, in case the gtf file does not have this information
    :return: the cleaned gtf df, such that all the required fields are present. columns: chromosome, source, feature, start, end, txrate, time, sequence, is_intron, intron_h
    """
    gtf_df = pd.read_csv(gtf_fn, sep='\t', header=0)  # chomosome, source, feature, start, end, txrate, time, sequence
    gtf_df['start'] = gtf_df['start'] - 1  # change the start coordinate to be 0-based, and so the coordinate of each feature is [start, end)
    try:
        gtf_df.drop('time', axis=1, inplace=True)   # drop the time column because we will calculate the time to traverse each feature based on the txrate later
    except:
        pass
    gtf_df['is_intron'] = gtf_df['feature'].apply(lambda x: x.split('_')[0]=='intron')  # exon_1, intron_1 etc. --> True or False whether this is an intron or not
    if 'intron_h' not in gtf_df.columns:
        gtf_df['intron_h'] = gtf_df['is_intron'].apply(lambda x: intron_h if x else 0)  # if this is an intron, then assign the intron_h, otherwise assign 0 (exon or irrelebant feature to splicing)
    assert 'txrate' in gtf_df.columns, 'txrate column is not in the gtf file'
    # now, assign PAS half life
    if gtf_df['feature'].iloc[-1] != 'PAS' or gtf_df['feature'].iloc[-1] != 'RTR':
        gtf_df = gtf_df._append({'feature': 'PAS', 'start': gtf_df['end'].iloc[-1], 'end': gtf_df['end'].iloc[-1] + 1, 'txrate': gtf_df['txrate'].iloc[-1], 'sequence': 'A', 'is_intron': False, 'intron_h': 0}, ignore_index=True)
        gtf_df = gtf_df._append({'feature': 'RTR', 'start': gtf_df['end'].iloc[-1], 'end': gtf_df['end'].iloc[-1] + RTR, 'txrate': gtf_df['txrate'].iloc[-1], 'sequence': 'A', 'is_intron': False, 'intron_h': 0}, ignore_index=True)
    elif gtf_df['feature'].iloc[-1] == 'PAS':
        gtf_df = gtf_df._append({'feature': 'RTR', 'start': gtf_df['end'].iloc[-1], 'end': gtf_df['end'].iloc[-1] + RTR, 'txrate': gtf_df['txrate'].iloc[-1], 'sequence': 'A', 'is_intron': False, 'intron_h': 0}, ignore_index=True)
    gtf_df['is_PAS'] = gtf_df['feature'].apply(lambda x: x=='PAS')
    if 'PAS_h' not in gtf_df.columns:
        gtf_df['PAS_h'] = gtf_df['is_PAS'].apply(lambda x: PAS_h if x else 0)  # if this is a PAS, then assign the PAS_h, otherwise assign 0 (irrelebant feature to cleavage)
    # add the readthrough region
    # TODO: modify this function to allow for multiple PAS sites
    gtf_df['time'] = (gtf_df['end'] - gtf_df['start']) / gtf_df['txrate'] / helper.ONE_KB  # time to traverse the feature
    return gtf_df

def get_reads_tagged_time(row, num_timepoints):
    """
    Given a row in the reads_df, determine the tagged_time of the read
    :param row: a row in the reads_df with required columns: abs_end, 0, 1, ..., num_timepoints-1 --> columns showsing the endpoint of the transcript at each time point
    Important assumption: columns 0,1,... num_timepoints-1 correspond to experiments done in chronoligcal time order
    :param num_timepoints: number of time points in the experiment
    :return: the tagged_time of the read. This corresponds to the time_idx of the experiment that the read will be assigned to if they are sequenced and mapped as part of the SLAM-seq pipeline
    """
    for time_idx in range(num_timepoints):
        if row['abs_end'] <= row[time_idx]:
            return time_idx
    return np.nan

def calculate_breakpoints_weilbull_fragment(trans_length, eta_val=200, avg_frag_len=250):
    """
    Given the length of the transcript and the number of fragments that we want to break the transcript into, this function will return a list showing the start point of each fragment in the transcript
    :param trans_length: transcript length (int)
    :param n: number of fragments (int)
    :param delta: parameters for the Wilbull distribution (float), log10(trans_length)
    :return: list of lengths and start points of each fragment (array)
    """
    delta = np.log10(trans_length)  # the log10 of the length of the transcript
    n = np.round(trans_length / (eta_val * gamma(1 + 1 / delta))).astype(
        int)  # num_fragments, based on the formulation of  https://academic.oup.com/nar/article/40/20/10073/2414449, foundation: Weibull distribution
    n = 1 if n <= 0 else n  # if n is negative, set it to 1
    relative_length = get_unif_fragment_relative(n)  # get the relative length of each fragment
    relative_length = np.power(relative_length, 1/delta)
    relative_length = relative_length / np.sum(relative_length)
    frag_length = np.round(trans_length * relative_length, 0)
    frag_start = np.concatenate((np.array([0]), frag_length))
    frag_start = np.cumsum(frag_start)[:-1]  # the last element is the length of the transcript, which is not the start of any fragment
    frag_length[-1] = trans_length-frag_start[-1]  # the last fragment may be slightly longer than the rest of the fragments because of rounding, so we fix it by trimming so that it ends at the end of the transcript
    return frag_length, frag_start


def adjust_absolute_read_coords(reads_df, intron_read_start, splicedO_b4_reads):
    """
    Break reads that overlap with a spliced out intron into two reads, one before the intron and one after the intron.
    :param reads_df: dataframe of reads. required columns: read_start, read_end, overlap_splicedI, precedeI. overlap_splicedI: whether the read overlaps with a spliced out intron. precedeI: the index of the spliced out intron BEFORE the read (indices compatible with the list intron_read_start and splice_out_len)
    :param intron_read_start: a list of length N+2, where N: number OF introns in the transcript. entry --> the start of each intron in the transcript, relative to the start of the transcript (0). It will be [0, trans_spl_df.start_within_trans, transcript_length]. The start_within_trans column shows the coordinate within the transcript that the start of EACH Íintron is located.
    :param splicedO_b4_reads: a list of length N+2, where N: number OF introns in the transcript. entry --> length of culmulative spliced out intron lengths UP UNTIL READS THAT START AFTER THE START OF THE INTRON
    :return: a dataframe, rows: reads, with the reads that overlap with a spliced out intron broken down into two reads, one before the intron and one after the intron.
    """
    break_reads = reads_df[reads_df['overlap_splicedI'] == True] # reads that overlap with a spliced out intron
    abs_reads = pd.DataFrame(columns=reads_df.columns)  # dataframe of reads with absolute coordinates
    if break_reads.shape[0] > 0:
        for row_idx, row in break_reads.iterrows():
            abs_reads.loc[abs_reads.shape[0]] = pd.Series({'read_start': row['read_start'], 'read_end': intron_read_start[row['precedeI']+1], 'trans_idx': row['trans_idx'], 'precedeI': row['precedeI']})  # the read before the spliced intron
            abs_reads.loc[abs_reads.shape[0]] = pd.Series({'read_start': intron_read_start[row['precedeI']+1], 'read_end': row['read_end'], 'trans_idx': row['trans_idx'], 'precedeI': row['precedeI']+1})  # the read after the spliced intron
    abs_reads= abs_reads._append(reads_df[reads_df['overlap_splicedI'] == False], ignore_index=True) # append the reads that do not overlap with a spliced out intron
    splicedO_b4_reads = pd.Series(splicedO_b4_reads).iloc[abs_reads['precedeI'].astype(int)] # the spliced out length of the introns BEFORE the read. each entry corresponds to a read (indices compatible with the list intron_read_start and splice_out_len)
    splicedO_b4_reads.reset_index(drop=True, inplace=True)
    abs_reads['abs_start'] = abs_reads['read_start'] + splicedO_b4_reads # calculate the absolute start of the read on the gene
    abs_reads['abs_end'] = abs_reads['read_end'] + splicedO_b4_reads  # calculate the absolute end of the read on the gene
    abs_reads.drop(['read_start', 'read_end', 'overlap_splicedI'], axis=1, inplace=True)
    abs_reads = abs_reads.sort_values('abs_start').reset_index(drop=True)  # sort the reads by their absolute start position
    # convert all columns to int
    abs_reads = abs_reads.astype(int)
    return abs_reads

def old_generate_reads(trans_lengths, eta_val=200, insertsize_min=200, insertsize_max=300, read_length = helper.DFT_READ_LENGTH):
    """
    Generate fragments from each transcript, size select, and return the starting position of the resulting
    reads relative to the length of the transcript. This function takes in the length of the transcripts and generate the fragments.
    Length of transcripts should have been calculated to excluse/include the intron regions as needed. This simply returns the start position of the fragments,
    with respect to the start of the transcript.
    :param trans_lengths: the length of the transcript, a vector of size (num_transcript)
    :param eta_val: eta value input to the Weibull distrubtion
    :param insertsize_min: minimum length of selected fragement (size-select). if -1--> no fragment size selection
    :param insertsize_max: maximum length of selected fragement (size-select). if -1--> no fragment size selection
    :param helper.DFT_READ_LENGTH: length of a read, for each fragment we will select the first helper.DFT_READ_LENGTH bp as a read. The rest of the fragment will be discarded in the sequencing process. if -1--> no portion of the transcripts will be thrown out and each read will actually be set to be the whole fragment
    :return: a dataframes showing the fragment start within each transcript. columns ['trans_idx', 'rel_start', 'length'] --> rel_start is the relative start of the fragment with respect to the start of the transcript. length is the length of the fragment.
    """
    num_transcript = len(trans_lengths)
    deltas = np.log10(trans_lengths)
    n = np.round(trans_lengths / (eta_val * gamma(1+ 1/trans_lengths))).astype(int)  # num_fragments, based on the formulation of  https://academic.oup.com/nar/article/40/20/10073/2414449, foundation: Weibull distribution
    n = np.where(n <= 0, 1, n)  # array of number of fragments for each transcript --> (num_transcript,)
    # n stands for the number of fragments that we will break the transcript into
    breakpoint_stats_list = list(map(lambda x: calculate_breakpoints_weilbull_fragment(trans_lengths[x], n[x], deltas[x]), range(num_transcript)))
    frag_length = list(map(lambda x: x[0], breakpoint_stats_list))  # list of arrays, each array is the length of each fragment in the transcript
    frag_start = list(map(lambda x: x[1], breakpoint_stats_list))  # list of arrays, each array is the start point of each fragment in the transcript, with the start point of the transcript being 0. Later functions need to calculate the absolute start point of the fragment on the gene by incorporating the gene_start and splicing status of transcripts
    # apply function to calculate the fragments generated from each transcript
    # --> list of arrays. Outter list: each transcript. Inner array: length of each fragment in the transcript
    df = pd.DataFrame()
    transcript_indices = np.repeat(range(len(trans_lengths)), n)
    df['trans_idx'] = transcript_indices
    df['read_start'] = np.concatenate(frag_start)
    df['length'] = np.concatenate(frag_length)
    if insertsize_min != -1 and insertsize_max != -1: # if either of them is -1, that means users do not want to generate reads but rather want to keep the whole transcripts, so we break the transcripts into fragments but we do not get rid of any part of any fragment
        df = df[(df['length']>= insertsize_min) & (df['length']<= insertsize_max)] # size select the fragment
    # fourth, for each selected fragment, obtain the reads. Each fragment generate one read.
    if read_length != -1:  # if users sepecify read_length as -1 that means they do not want to get rid of any portion of the trasncripts (we can break the transcripts into fragments, but we will not take the first read_length bp of each fragment as reads. Instead, we will keep the whole fragment as a read)
        df['read_end'] = df['read_start'] + read_length  # calculate the end of the read
    else:
        df['read_end'] = df['read_start'] + df['length']
    df.drop(['length'], axis=1, inplace=True)   # redundant column, replaced by read_start
    return df

def old_generate_reads_from_transDF(trans_df, trans_spl_df, gtf_df, eta_val=200, insertsize_min=200, insertsize_max=300, read_length=helper.DFT_READ_LENGTH):
    """
    This function is  old and is no longer relevant, but dont want to get rid of it yet in case I need it for future reference.
    Generate fragments from each transcript, size select, and return the starting position of the resulting
    reads relative to the length of the transcript. This function takes in the dataframe of transcripts simulated at each time point
    :param trans_df: columns outlining the transcript information. required columns: trans_idx, end_site, cleaved. Other columns are unrelated to this function. If cleaved =1, the end_site is at the cleavage site (PAS in the gtf_df).
    :param trans_spl_df: columns outlining the splicing and cleavage status of the transcripts. Required column: feature, intron_h, is_intron, spliced
    :param gtf_df: columns outlining the gtf information. Required columns: feature, start, end, strand
    :param eta_val: eta value input to the Weibull distrubtion
    :param insertsize_min: minimum length of selected fragement (size-select)
    :param insertsize_max: maximum length of selected fragement (size-select)
    :param helper.DFT_READ_LENGTH: length of the read, for each fragment we will select the first helper.DFT_READ_LENGTH bp as a read. The rest of the fragment will be discarded in the sequencing process
    :return: a dataframe showing the read start ABSOLUTE coordinate. columns ['trans_idx', 'start', 'end'] --> each row is a read, and if a read is a junction read, it will be broken down into two reads, one before the junction and one after the junction --> number of rows will be larger than number of fragments.
    This output can be later used to calculate the read coverage along the gene.
    """
    # first, calculate the real length of the transcripts given the splicing status
    gene_start = gtf_df['start'].iloc[0]  # the start of the gene
    splice_out_len_per_trans = trans_spl_df.groupby('trans_idx').apply(lambda x: (x['is_spliced'] * x['Ilen']).sum())  # for each transcript, calculate the length of the introns that are spliced out (intron length * whether or it was spliced out)
    # --> a pandas series with index: trans_idx, and value: length of the introns that are spliced out
    trans_df['length'] = trans_df['end_site'] - gene_start - splice_out_len_per_trans  # calculate the length of the transcript
    # second, calculate fragment starts, ends and lengths given the real length of the transcripts
    reads_df = old_generate_reads(trans_df['length'], eta_val=eta_val, insertsize_min=insertsize_min, insertsize_max=insertsize_max, read_length=helper.DFT_READ_LENGTH)  # generate fragments, df with columns: ['trans_idx', 'rel_start', 'length'] each row is a fragment belonging to a transcript. rel_start is always with respect with to the start of the transcript, we will have to adjust the absolute start position of the fragment later based on the splicing status of the transcript and gene_start
    # Note the function called above already size-selected the fragments --> which has the potential to throw a bunch of fragments, but I assume it's realistic simulation of the sequencing process
    # Also because of size-selection, we may end up with cases where there are no fragments in a transcript because it's possible the transcript is too short --> 1 fragment--> size-selected out
    # Note: until here, reads' start and end are relative to the start of the transcript (which is 0 here, but the absolute transcript start is the gene_start coordinate)
    print('Done generating raw reads. Now, mapping them onto the gene based on splicing status of transcripts')
    """
    Up until here, below is an example trans_spl_df:
    feature  start     Ilen  intron_h  txrate  Udist  trans_idx  end_site  splice_prob  is_spliced
0  intron_1    343  62805.0         5     2.9    329          0     76821     0.480185           1
1  intron_2  63299   3600.0         5     2.9  63285          0     76821     0.378098           0
2  intron_3  67108   6094.0         5     2.9  67094          0     76821     0.159425           0
3  intron_4  73377   9263.0         5     2.9  73363          0     76821     0.000000           0
start is the ABSOLUTE intron start site 

    below is an example reads_df:
    trans_idx  rel_start  read_end  read_start
0           0        0.0      50.0         0.0
1           0      271.0     321.0       271.0
5           0     1077.0    1127.0      1077.0
12          0     2375.0    2425.0      2375.0
14          0     2694.0    2744.0      2694.0
    """
    # first, calculate the coordinates within the transcript length of the start of each intron
    # in each transcript, for each intron, we calculate: if the beginning of transcript is 0, then what is the coordinate of the start of the intron within the transcript length (given that some introns are spliced out and some are not)
    trans_spl_df['spliced_out_len'] = trans_spl_df['Ilen'] * trans_spl_df['is_spliced']
    # let BLEN  denote the length of the introns that are spliced out up until the BEGINNING of the current intron (in the current row) of each transcript. Therefore, for the first intron in each transcript, this value will be 0
    trans_spl_df['Blen'] = trans_spl_df.groupby('trans_idx')['spliced_out_len'].transform(lambda x: x.cumsum().shift(fill_value=0))  # calculate the cumulative length of the introns that are spliced out up until the BEGINING of the current intron (in the current row) of each transcript.
    trans_spl_df['start_within_trans'] = trans_spl_df['start'] - gene_start - trans_spl_df['Blen']  # calculate the start of each intron within the transcript length
    # second, calculate the number of spliced out introns in each transcript
    trans_spl_grouped = trans_spl_df.groupby('trans_idx')  # each transcript is a group  --> splicing status of the transcript --> each row is an intron
    reads_grouped = reads_df.groupby('trans_idx')
    # third, loop through each transcript and find the SPLICED OUT intron break BEFORE each read, and whether the read overlaps with the SPLICED out intron. Note, we only care about the spliced out introns because we will only need to adjust the reads' ABSOLUTE coordinates by the 'spliced_out_len' of spliced out introns.
    adj_frag_df_list = [] # list of dataframes, each dataframe is the df of reads in a transcript, but with the adjusted coordinates (absolute coordinates on teh genome), any reads that overlap with a spliced out intron will be broken into two reads, one before the intron and one after the intron
    for trans_idx, trans_spl in trans_spl_grouped:
        trans_spl = trans_spl.copy() # df showing the splicing status of the transcript, each row is an intron
        try:
            reads_in_trans = reads_grouped.get_group(trans_idx).copy()  # df, rows: reads in transcripts, required columns: read_start, read_end
        except:  # no reads in this transcript because of size-selection throwing out all fragments
            continue
        trans_len = trans_df.loc[trans_idx, 'length']  # length of this transcript
        # get the start of SPLICED OUT introns in this transcript (plus the first location 0 and last location as the length of the transcript)
        intron_read_start = [0] + list(trans_spl['start_within_trans']) + [trans_len]
        # get the splicing status of each intron in this transcript (plus the first and last intron which are not spliced out)
        is_spliced = [0] + list(trans_spl['is_spliced']) + [0]
        # get the culmulative spliced out length until the END of each intron in this transcript (plus first: 0, last: last intron's spliced out length, beacuse they correspond to transcript start and end)
        splicedO_b4_reads = trans_spl['Blen']+trans_spl['is_spliced']*trans_spl['Ilen']
        splicedO_b4_reads = [0] + list(splicedO_b4_reads) + [splicedO_b4_reads.iloc[-1]] # culmulative length of spliced out introns until the END of each intron in this transcript. First and last "intron" here correspond to the start and end of transcript, so their spliced out length is 0  and spliced_out_len[-1]
        # we will find the intron index that immediately precedes the read, and splicedO_b4_reads will be the spliced out length that immediately precedes the read. Following, we will adjust the
        reads_in_trans[['overlap_splicedI', 'precedeI']] = reads_in_trans.apply(lambda row: check_intron_break_per_read(row, intron_read_start, is_spliced), axis=1) # for each read, this function will return two values: (1) whether the read overlaps with a spliced out intron, (2) the index of the spliced out intron BEFORE the read
        # now we will adjust the reads so that they have the absolute start and end coordinate
        adj_frag_df = adjust_absolute_read_coords(reads_in_trans, intron_read_start, splicedO_b4_reads) # colums: trans_idx, abs_start, abs_end, precedeI
        adj_frag_df_list.append(adj_frag_df)
    adj_frag_df = pd.concat(adj_frag_df_list, ignore_index=True)  # all reads from all transcripts
    # finally, add gene_start to all the coordinate
    adj_frag_df['abs_start'] = adj_frag_df['abs_start'] + gene_start
    adj_frag_df['abs_end'] = adj_frag_df['abs_end'] + gene_start
    return adj_frag_df

