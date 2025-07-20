import pandas as pd
import numpy as np
import helper
from scipy.special import gamma
# BUFFER_JUNCTION=9  # number of bf before the intron start site that we will consider a read to be a junction read.
# In Athma's simulation, for a read length helper.DFT_READ_LENGTH to be considered a junction read, the start of the read must be within (helper.DFT_READ_LENGTH-BUFFER_JUNCTION-1, BUFFER_JUNCTION) bp of ahead the intron start site
# However, here I define junction simply as junction reads (overlapping with part of the intron or skipping the intron), so I will not use this BUFFER_JUNCTION parameter ** for now **
DEFAULT_SEED = 9999

def set_seed(seed=DEFAULT_SEED):
    """
    Set the seed for the random number generator
    :param seed: seed for the random number generator
    :return: None
    """
    np.random.seed(seed)


##### this will set the seed for the random number generator
set_seed()

def get_unif_fragment_relative(n):
    '''
    This function will take in n as the number of fragments, and will return an array of numbers  of length n
    The sum of numbers will be 1, corresponding to the relative length of each fragment within the transcript/molecule
    Different functions will apply different trnasformation to these relative lengths to get the absolute lengths of the fragments
    :param n: number of fragments
    :return:
    '''
    relative_break = np.random.uniform(0, 1, n - 1)  # number of breakpoints is number of fragments - 1
    relative_break = np.sort(relative_break)  # [0.3,0.4]
    relative_break = np.concatenate((np.array([0]), relative_break, np.array([1])))  # [0, 0.3, 0.4, 1]
    relative_length = np.diff(relative_break)  # [0.3, 0.1, 0.6] length n, all values sum to 1
    return relative_length

def calculate_breakpoints_uniform_fragment(trans_length, eta_val = 200, avg_frag_len= 250):
    """
    eta_val would not be used in this function, but we keep it to make the format for the two fragmentation methods consistent
    return frag_length, frag_start
    The simulation will try to break down a transcript into fragments of average length avg_frag_len, using uniform fragmentation
    The procedure:
    - Given the length of the transcript and the expected avg_frag_len of a fragment --> n number of fragment is drawn from Poisson distribution
    - The relative length of each fragment is drawn from a uniform distribution --> draw n numbers that sum to 1
    - We will specifically try to mimic the fragmentation process in which the first break point is chosen randomly
    - Then, we will simulate other breakpoints from the first breakpoint and the relative length of each fragment
    The point of this procedure is that we want to mimic uniform fragmentation, but we also want to simualte the edge effects of the fragmentation process
    In which the fragments associated with the beginning of the gene tends to be shorter than the fragments associated with the middle of the gene
    """
    n = np.random.poisson(trans_length/avg_frag_len)  # number of fragments
    if n == 0:  # if n is 0, we will have to set n to 1, because we need at least one fragment
        n = 1
    relative_length = list(get_unif_fragment_relative(n))  # get the relative length of each fragment, length n
    # choose a random place that is considered first break point
    first_break = np.random.uniform(0,1)  # 0 or 1
    break_coord_list = [first_break]  # list of break points. We will add the breakpoints before this break to the beginning of the list, and add breakpoints after this break to the end of the list
    # from the first, calculate the other breakpoint to the beginning of the gene
    Udist = first_break # distance from the TSS to the current break point
    while Udist > 0:  # this while looop is design such that the fragments are added until we have left a shorter fragment at the beginning of the gene
        try:
            next_frag = relative_length.pop(0)  # remove the first element from the list as the fragment length of the next fragment
            Udist -= next_frag
            break_coord_list.insert(0, Udist)  # add the break point to the beginning of the list
        except: # relative_length is empty, which should not happen at all
            break
    Ddist = 1 - first_break  # distance from the TES to the current break point
    while Ddist > 0:
        try:
            next_frag = relative_length.pop()  # remove the last element from the list as the fragment length of the next fragment
            Ddist -= next_frag
            break_coord_list.append(1-Ddist)  # add the break point to the end of the list
        except: # relative_length is empty, which may happen
            break
    # we may end up with fragmentation such that the first fragment's coord is <0. Clip it to 0
    break_coord_list = np.clip(break_coord_list, 0, 1)  # length n, each corresponds to the start of a fragment
    # convert the relative break into absolute break
    frag_start = np.round(trans_length * break_coord_list, decimals=0) # length n, each corresponds to the start of a fragment
    frag_length = np.append((np.diff(frag_start)), trans_length - frag_start[-1])  # length n, each corresponds to the length of a fragment
    return frag_length, frag_start

def calculate_breakpoints_weilbull_fragment(trans_length, eta_val=200, avg_frag_len=250):
    """
    Given the length of the transcript and the number of fragments that we want to break the transcript into, this function will return a list showing the start point of each fragment in the transcript
    :param trans_length: transcript length (int)
    :param n: number of fragments (int)
    :param delta: parameters for the Wilbull distribution (float), log10(trans_length)
    :return: list of lengths and start points of each fragment (array)
    """
    delta = np.log10(trans_length)  # the log10 of the length of the transcript
    n = np.round(trans_length / (eta_val * gamma(1 + 1 / delta))).astype(int)  # num_fragments, based on the formulation of  https://academic.oup.com/nar/article/40/20/10073/2414449, foundation: Weibull distribution
    n = 1 if n <= 0 else n  # if n is negative, set it to 1
    relative_length = get_unif_fragment_relative(n)  # get the relative length of each fragment
    relative_length = np.power(relative_length, 1/delta)
    relative_length = relative_length / np.sum(relative_length)
    frag_length = np.round(trans_length * relative_length, 0)
    frag_start = np.concatenate((np.array([0]), frag_length))
    frag_start = np.cumsum(frag_start)[:-1]  # the last element is the length of the transcript, which is not the start of any fragment
    frag_length[-1] = trans_length-frag_start[-1]  # the last fragment may be slightly longer than the rest of the fragments because of rounding, so we fix it by trimming so that it ends at the end of the transcript
    return frag_length, frag_start

def break_fragment_into_two(max_frag_len):
    if max_frag_len>0:
        frag1_len = np.random.randint(0, max_frag_len)
    else:
        frag1_len = 0
    return frag1_len, max_frag_len - frag1_len

def calculate_breakpoints_with_edge_effects(trans_length, eta_val=200, avg_frag_len=250):
    """
    This function will calculate the breakpoints of the fragments of a transcript, given the length of the transcript and the average length of the fragments
    :param: trans_length: length of the transcript
    :param: eta_val: scale parameter for the Weibull distribution
    :param: avg_frag_len: average length of the fragments. We wont use it in this function, but we keep it to make the format for the two fragmentation methods consistent
    """
    # first get the frag_length and frag_start with no edge effects
    frag_length, frag_start = calculate_breakpoints_weilbull_fragment(trans_length, eta_val=eta_val, avg_frag_len=avg_frag_len)
    # now, simulate the edge effects:
    # - First simulate the maximum length possible of the first fragment's endpoint. We assume it's the same as the Weibull distirbution used to simulate fragment length again
    firstFrag_len, secondFrag_len = break_fragment_into_two(frag_length[0])
    lastFrag_len, secondLastFrag_len = break_fragment_into_two(frag_length[-1])
    # add the data of the first fragment to the beginning of the frag_length and frag_start
    frag_length = np.insert(frag_length, 0, firstFrag_len)
    frag_length[1] = secondFrag_len  # adjust the length of the second fragment
    if len(frag_length) > 2: # if the transcript has more than 1 fragments in the first fragmentation (before breaking the first fragment into two)
        # this if statement is needed because we want to avoid the case when the transcript is very short and can be broker into only 1 fragment from WeiBull distribution
        # In this case, the transcript is long enough, so both the first and the last fragment can be broken into two
        frag_length[-1] = secondLastFrag_len
        frag_length = np.append(frag_length, lastFrag_len)
    frag_start = np.cumsum(frag_length)[:-1]
    frag_start = np.insert(frag_start, 0, 0)
    return frag_length, frag_start

def check_intron_break_per_read(row, intron_read_start, is_spliced):
    """
    Determine if a read needs to be broken down into two reads (overlap a spliced region), based on the start and end of the read and the start of the introns in the transcript. the start of the introns are relative to the start of the transcript (0), and also takes into account whether the introns are spliced out or not.
    :param row: a row of the reads_df, which corresponds to a read. required columns: read_start, read_end. read_start and read_end are relative to the start of the transcript (0) and until the length of the transcript. The length of the transcript is calculated based on the splicing of individual introns of the transcript.
    :param intron_read_start: a list of length N+2, where N: number OF introns in the transcript. entry --> the start of each intron in the transcript, relative to the start of the transcript (0). It will be [0, trans_spl_df.start_within_trans, transcript_length]. The start_within_trans shows the coordinate within the transcript that the start of EACH Íintron is located.
    :param: is_spliced: a list of length N+2, where N: number OF introns in the transcript. entry: whether (1) or not (0) the intron is spliced out. The first and last entry of this list is 0 (unspliced) because they correspond to the start and end of the transcript -- not a particular intron.
    :return: (1) True or False, whether the read needs to be broken down into two reads (one part before an intron and one part after the intron).
    (2) the index of the intron that starts BEFORE the read starts. Note that the index is with respect to the ordering in the input list intron_read_start
    """
    if len(intron_read_start) == 2:  #this gene has no potential introns, two numbers are [0, transcript_length]
        return False, 0  # no need to break down the read
    assert intron_read_start[0] <= row['rel_start'], 'the read coordinate is not within the transcript length, start: {}, end: {}'.format(row['rel_start'], row['rel_end'])
    assert intron_read_start[-1] >= row['rel_end'], 'the read coordinate is not within the transcript length, start: {}, end: {}'.format(row['rel_start'], row['rel_end'])
    int_idx = 0
    int_start = intron_read_start[int_idx]
    while int_start <= row['rel_start']:
        int_idx += 1
        int_start = intron_read_start[int_idx]
    # up until here, int_idx is the intron index that starts AFTER the read starts
    # we have to check if the intron start is before the read ends. if before --> break the read, if not --> don't break the read
    if int_start < row['rel_end'] and is_spliced[int_idx]==1:  # this read starts before the intron start but ends after the intron start
        # we do not accept the case int_start == row['read_start'] because reads are denoted [start, end), so if the read ends at the intron start, it is not overlapping with the intron
        return pd.Series({'overlap_splicedI':True, 'precedeI':int_idx-1})
    else: # if the read ends after the intron start, but the intron is not spliced out, then we don't need to break the read. Or, if the read ends before the intron start, then we don't need to break the read
        return pd.Series({'overlap_splicedI':False, 'precedeI':int_idx-1})

def adjust_absolute_one_read_coords(row, splice_df, gene_start):
    """
    This function will adjust one read at a time,
    :param row: one row in the reads_df, corresponding to one read and its information. required columns: read_start, read_end, overlap_splicedI, precedeI.
    overlap_splicedI: whether the read overlaps with a spliced out intron. precedeI: the index of the spliced out intron BEFORE the read starts (indices compatible with the list intron_read_start and splice_out_len)
    :param intron_read_start: a list of length N+2, where N: number OF introns in the transcript. entry --> the start of each intron in the transcript, relative to the start of the transcript (0). It will be [0, trans_spl_df.start_within_trans, transcript_length]. The start_within_trans shows the coordinate within the transcript that the start of EACH Íintron is located.
    :param: is_spliced: a list of length N+2, where N: number OF introns in the transcript. entry: whether (1) or not (0) the intron is spliced out. The first and last entry of this list is 0 (unspliced) because they correspond to the start and end of the transcript -- not a particular intron.
    :param: gene_start: the ABSOLUTE start of the gene on the chromosome
    :return: abs_start, abs_end, subtract_start, subtract_end --> absolute start and end of the read on the gene, and ABSOLUTE start and end of spliced out regions that this read overlap (if this reads overlap with a spliced out intron then subtract_start and subtract_end should corrrespond to the ABSOLUTE start and end of the intron
    """
    if splice_df.empty:  # this gene has no potential introns
        return pd.Series({'abs_start': row['rel_start']+gene_start, 'abs_end': row['rel_end']+gene_start, 'subtract_start': 0, 'subtract_end': 0})
    precedeI = row['precedeI']-1 # index of the intron that STARTS before the read starts ( this intron can be spliced out or not, but it starts before our read starts)
    if precedeI == -1:  # this read starts before the first intron starts
        preI_abs_start = gene_start
        preI_rel_start = 0
        spliced_out_len = 0
    else:
        preI_abs_start = splice_df.loc[precedeI, 'start']  # absolute start of the intron that starts before the read starts
        preI_rel_start = splice_df.loc[precedeI, 'start_within_trans'] # relative start of the intron that starts before the read starts
        spliced_out_len = splice_df.loc[precedeI, 'spliced_out_len']  # length of the intron that starts before the read starts, if it is spliced out
    if row['overlap_splicedI'] == False:  # this read does not overlap with a spliced out intron
        abs_start = (row['rel_start']-preI_rel_start) + preI_abs_start + spliced_out_len  # absolute start of the read on the gene
        abs_end = (row['rel_end']-preI_rel_start) + preI_abs_start + spliced_out_len # absolute end of the read on the gene
        subtract_start = 0
        subtract_end = 0
    else:  # this read overlaps with a spliced out intron
        abs_start = (row['rel_start']-preI_rel_start) + preI_abs_start + spliced_out_len  # absolute start of the read on the gene
        abs_end = (row['rel_end']-splice_df.loc[precedeI+1, 'start_within_trans']) + splice_df.loc[precedeI+1, 'end']  # absolute end of the read on the gene
        subtract_start = splice_df.loc[precedeI+1, 'start']
        subtract_end = splice_df.loc[precedeI+1, 'end']
    return pd.Series({'abs_start': abs_start, 'abs_end': abs_end, 'subtract_start': subtract_start, 'subtract_end': subtract_end})


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

def get_pairedEnd_reads_df(reads_df):
    """
    Given a dataframe of reads, generate a dataframe of paired-end reads. Each read will be paired with the next read in the reads_df.
    :param reads_df: dataframe of reads that are at the beginning of the fragment
    required columns: rel_start, rel_end, frag_start, frag_len, frag_idx
    """
    other_end_reads = reads_df.copy()  # columns: rel_start, frag_len, frag_idx, rel_end, read_len
    middle_reads = reads_df.copy()
    other_end_reads['rel_end'] = other_end_reads['rel_start'] + other_end_reads[ 'frag_len']  # calculate the end of the read
    other_end_reads['rel_start'] = other_end_reads['rel_end'] - other_end_reads['read_len']
    middle_reads['rel_start'] = reads_df['rel_end']
    middle_reads['rel_end'] = middle_reads['rel_start']
    middle_reads['read_len'] = middle_reads['rel_end'] - middle_reads['rel_start']
    middle_reads = middle_reads[middle_reads['read_len'] > 0]  # remove reads with negative length
    reads_df['pairedEnd_type'] = 'first'
    other_end_reads['pairedEnd_type'] = 'second'
    middle_reads['pairedEnd_type'] = 'middle'
    reads_df = pd.concat([reads_df, other_end_reads, middle_reads], ignore_index=True)
    return reads_df

# def check_junction_type(row):
#     """
#     Given a read, determine if it is an exon-exon junction read or an exon-intron junction read
#     :param row: a read. required columns start_feat, end_feat: features that the start and end point of the reads overlap with
#     :return: the type of function reads and the intron that it is associated with
#     """
#     # TODO: Right now this function assumes that a junction read cannot span over two different introns (which may correspond to alternative splicing event)
#     # TODO: This function also assumes that the read length is short enough that it cannot span >2 exons/ introns
#     [start_type, start_ftI] = row['start_feat'].split('_') # start_type: exon or intron. start_ftI: feature index (0-based), exon_0 --> exon, 0th (exon)
#     [end_type, end_ftI] = row['end_feat'].split('_') # end_type: exon or intron. end_ftI: feature index (0-based), exon_0 --> exon, 0th (exon)
#     start_ftI = int(start_ftI)
#     end_ftI = int(end_ftI)
#     if start_type == 'exon' and end_type == 'exon' and start_ftI == end_ftI-1:  # exon-exon junction read
#         return 'EE', start_ftI
#     if start_type == 'exon' and end_type == 'intron' and start_ftI == end_ftI: # exon-intron junction read but at the 5' end of the intron
#         return 'EI', start_ftI
#     if start_type == 'intron' and end_type == 'exon' and start_ftI == end_ftI+1: # exon-intron junction read but at the 3' end of the intron
#         return 'IE', start_ftI
#     return 'None', -1
#
# def calculate_junction_reads(reads_df, gtf_df):
#     """
#     This function will calculate, for each intron, the number of reads that are considered junction reads.
#     There are two types of junction reads of interest: exon-exon junction reads and exon-intron junction reads.
#     1 exon-exon junction read corresponds to 1 transcript from a spliced transcript
#     1 exon-intron junction read corresponds to 1 transcript from an unspliced transcript. Here, we should focus on exon-intron junction read that are at the
#     3' end of the intron, because 1 read overlapping that region will correpond to 1 transcript that are elongated beyong the end of the intron, but are not spliced
#     TODO: check if this is a valid assumption
#     :param reads_df: Data of the reads from sequencing experiment. Each row corresponds to a read. required columns: abs_start, abs_end --> abosolute start and end point of the read [start, end)
#     :param gtf_df: data of gene features, sorted by order on the gene (each feature can be intron, exon or others). required columns: feature, start, end --> start and end of the gene feature [start, end)
#     :return:
#     """
#     # first, order the gtf_df by the start position of each feature
#     gtf_df = gtf_df.sort_values('start').reset_index(drop=True)
#     feat_start = pd.Series(index = gtf_df.feature, data=gtf_df.start.values)  # start position of each feature
#     # for each read, find the feature the read start overlap with. This should be the last feature whose start is smaller than the read start
#     reads_df['start_feat'] = reads_df['abs_start'].apply(lambda x: feat_start[feat_start <= x].index[-1])
#     # for each read, find the feature the read end overlap with. This should be the last feature whose start is smaller than the read end
#     reads_df['end_feat'] = reads_df['abs_end'].apply(lambda x: feat_start[feat_start <= x].index[-1])
#     # based on features overlapping read start and read end, determine if the read is a junction read
#     junc_type, intron_idx = zip(*reads_df.apply(lambda row: check_junction_type(row), axis=1)) # junc_type: type of junction read EE IE EI, intron_idx: index of the intron that the read is associated with
#     reads_df['junc_type'] = junc_type
#     reads_df['intron_idx'] = intron_idx
#     # calculate the number of junction reads for each intron
#     ee = reads_df[reads_df['junc_type'] == 'EE'].groupby('intron_idx').size() # number of exon-exon junction reads for each intron --> index: intron_idx, value: number of reads
#     ei = reads_df[reads_df['junc_type'] == 'EI'].groupby('intron_idx').size() # number of exon-intron junction reads for each intron --> index: intron_idx, value: number of reads
#     ie = reads_df[reads_df['junc_type'] == 'IE'].groupby('intron_idx').size() # number of exon-intron junction reads for each intron --> index: intron_idx, value: number of reads
#     return ee, ei, ie

