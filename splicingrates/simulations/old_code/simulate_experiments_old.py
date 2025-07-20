import pandas as pd
import numpy as np
import argparse
import helper

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

def determine_splice_per_transcript(transcript_index, end_site, gtf_df):
    """
    given the end site of a transcript, determine the splicing probability of each intron along the transcript
    :param transcript_index: the index of the transcript (indexed by the order in which the transcript was sampled). This is useful because for each transcript, there are multiple spliced sites --> multiple splicing probabilities --> multiple indicators of whether the transcript is spliced or not. We want to be able to keep track of all of this information
    :param end_site: the end site of the transcript, end_site was generated in the range (gene_start, maximum_transcript_length) in the function simulate_starting_transcripts or create_new_transcripts
    :param gtf_df:
    :return:
    """
    gene_start = gtf_df['start'].iloc[0]  # the start of the gene
    splicing_df = gtf_df[gtf_df['is_intron']][['feature', 'start', 'length', 'intron_h', 'txrate']]  # only consider the introns. We will eventually be interested in calculating, for each intron, the probability that THIS transcript (with THIS end_site) is spliced at each intron
    splicing_df['Udist'] = splicing_df['start'] - gene_start  # distance from TSS to the beginning of the intron
    splicing_df.reset_index(inplace=True, drop=True)  # reset the index of the dataframe
    splicing_df.rename(columns = {'length': 'Ilen'}, inplace=True)  # rename the columns to be consistent with the simulation framework
    # Ilen: length of intron
    # Udist: distance from TSS to the beginning of the intron
    splicing_df['trans_idx'] = transcript_index
    splicing_df['end_site'] = end_site
    splicing_df['splice_prob']= splicing_df.apply(lambda row: 1-2**(-(end_site-row['Udist']-row['Ilen'])/(row['intron_h']*row['txrate']*ONE_KB)), axis=1)
    splicing_df['splice_prob'] = splicing_df['splice_prob'].apply(lambda splice_prob: 0 if splice_prob<0 else splice_prob)
    # the probability that this transcript is spliced at this intron P(spliced|end_site=0, intron_h, txrate) = 0 and P(spliced|end_site=intron_h*txrate, intron_h, txrate) = 1/2, where end_site is relative to the END of the intron
    # so P(spliced| end_site) = 1- 2**{-\frac{x}{hr}}, where x is endsite RELATIVE TO INTRON END, and h is the half life of splicing in minutes and r is transcription rate in bp (not KB)/min
    # sample whether each intron has been spliced or not on this transcript given the splice probability
    splicing_df['is_spliced'] = np.random.binomial(1, splicing_df['splice_prob'])
    return splicing_df


def simulate_splicing_from_curr_trans(trans_df, gft_df):
    """
    simulate splicing of the transcripts that are currently in the cell. For each transcript, we only need to care about the end point of the transcript, because this is the only information that we need to determine the splicing status of each intron along the transcript
    :param trans_df: dataframe of the current transcripts, what we really care about to determine the splicing patterns is the endsite of the transcript (how far it has gone since the beginning of gene--TSS). columns: transcript, end_site, time_idx
    :param gft_df: gene annotation df. columns: chromosome, source, feature, start, end, length, txrate (in kb per minute), time, sequence, is_intron, intron_h
    :return: df showing the splicing status of each transcript in the input trans_df. columns: feature, Udist, Ilen, intron_h, txrate, trans_idx, end_site, splice_prob, is_spliced
    """
    # first we will pair the trans_idx with end_site
    idx_end_pair = zip(trans_df['trans_idx'], trans_df['end_site'])
    # next we will apply the splicing simulation function to each transcript
    splicing_df_list = list(map(lambda x: determine_splice_per_transcript(x[0], x[1], gft_df), idx_end_pair))  # list of df, each df shows the splicing status of each each transcript. Each row in the df corresponds to one intron in the gene. The transcript may or may not pass through each intron, but if the end point of the transcript is beyond the intron endpoint, then the transcript will be spliced at that intron with a certain probability >0.
    trans_spl_df = pd.concat(splicing_df_list, axis=0)  # concatenate all the splicing_df into one dataframe, so this df outlines the splicing status of all the transcripts sampled. Columns: feature, Udist, Ilen, intron_h, txrate, transcript, end_site, splice_prob, is_spliced
    return trans_spl_df

def simulate_cleavage_from_curr_trans(trans_df, gtf_df):
    """
    this function will determine whether each of the transcript is cleaved or not, given the cleavage half life and the transcripts' end_site
    :param trans_df: df of the current transcripts. Columns: trans_idx, end_site, time_idx
    :param gtf_df: gene annotation df. columns: chromosome, source, feature, start, end, length, txrate (in kb per minute), time, sequence, is_intron, intron_h
    :return:
    """
    # TODO: change this function to simulate cleavage event such that there can be multiple cleavage sites along the transcript. Right now, I am assuming that there is only one cleavage site along the transcript. This function will just determine whether the transcripts should be cleaved or not. If cleaved, change the end site of the transcript to be the cleavage site
    PAS_end_coord = (gtf_df[gtf_df['feature']=='PAS'])['end'].iloc[0]  # row of the RTR feature
    PAS_coord = (gtf_df[gtf_df['feature']=='PAS'])['start'].iloc[0]  # the genomic coordinate of the PAS start site (the end of the gene). It is possible that a gene has multiple PAS, and assume that features in gene are sorted in ascending order of their start site. Here, we find the first PAS in the gene, which is usually the constitutuve PAS. This function right now can only simulate the case of 1 PAS
    PAS_h = (gtf_df[gtf_df['feature']=='PAS'])['PAS_h'].iloc[0]  # cleavage half life, the first PAS in the gene
    txrate =(gtf_df[gtf_df['feature']=='PAS'])['txrate'].iloc[0]  # transcription rate of the PAS feature, the first PAS in the gene
    cleaved_prob = 1-2**(-(trans_df['end_site']-PAS_coord)/(PAS_h*txrate))
    # for each transcript, the probability that has been cleaved given its end site and the cleavage half life
    cleaved_prob = np.where(cleaved_prob < 0, 0, cleaved_prob)
    # if the probability is negative (endsites before the gene end), set it to 0 --> cannot be cleaved yet
    cleaved = (np.random.binomial(1, cleaved_prob)==1)
    trans_df['cleaved'] = cleaved
    trans_df['end_site'] = trans_df['cleaved'] * PAS_end_coord + (1-trans_df['cleaved']) * trans_df['end_site']  #if cleaved, then the end site is the end of the gene, otherwise, the end site is the same as before
    return trans_df

def simulate_starting_transcripts(gtf_df, init_exp, num_total_transcript_millions):
    """
    simulate the transcripts at t=0, with reads that have random end points
    - Estimate the splicing status of each transcript
    - Estimate the cleavage status of each transcript
    :param gtf_df: gene annotation file. chromosome, source, feature, start, end, length, txrate (in kb per minute), time, sequence, is_intron, intron_h
    :param init_exp: expression level of the gene at time t=0, in TPM. This may be changed in later implementation of the simulation because I am not sure if it is realisitic to assume that we know gene expression at t=0 since we are dealing with nascent mRNA.
    :param num_total_transcript_millions: number of millions of transcripts that this cell/cells generated through transcription
    :return: a dataframe of fragments from transcripts. columns: transcript, start, end, length, spliced, junction, read_name, flag, RNAME, POS, MAPQ, CIGAR, RNEXT, PNEXT, TLEN, SEQ, QUAL, TAG12345, CIGAR
    """
    num_transcripts_to_start = init_exp * num_total_transcript_millions
    # find the length of the gene and the maximum length possible of a transcript (assuming it can go beyond the end of the PAS)
    gene_start = gtf_df['start'].iloc[0]  # the start of the gene
    max_transcript_length = np.max(gtf_df['end'])  # due to how the function read_gtf_file is written, the last feature in gtf_df is the PAS and RTR, so the end of the last feature is the end of the gene plus the RTR length
    # randomly sample the end sites of the transcripts
    endsites = np.random.randint(gene_start, max_transcript_length, num_transcripts_to_start)
    # for each transcript, we will determine the actual transcript given the end site and the splice sites along this transcripts and the splicing halflife for each intron. Therefore, on each transcript, there can be multiple spliced sites and potentially multiple included introns, depending on the splicing halflife of each intron
    trans_df = pd.DataFrame({'time_idx':0, 'end_site':endsites}).reset_index(drop=False, inplace=False).rename(columns = {'index':'trans_idx'})  # df outlines all the transcripts that get generated at this time point. index of this df is exactly the index of the transcript
    # df outlines all the transcripts that get generated at this time point. index of this df is exactly the index of the transcript
    trans_spl_df = simulate_splicing_from_curr_trans(trans_df, gtf_df)  # df that outlines the splicing status of each transcript. columns: feature, Udist, Ilen, intron_h, txrate, transcript, end_site, splice_prob, is_spliced
    # note that Udist is the distance from the TSS to the beginning of the intron, and Ilen is the length of the intron, whether TSS is 0 or absolute coordinate of the gene
    trans_df= simulate_cleavage_from_curr_trans(trans_df, gtf_df)  # added column 'cleaved'
    return trans_df, trans_spl_df


def simulate_transcript_elongation(time_idx, time_since_prev, prev_trans_df, prev_trans_spl_df, gtf_df):
    """
    This function should simulate the elongation of the transcripts from the previous time point to this time point, given the transcripts that were created from previous time point. This function does not consider the creation of new transcripts, only the elongation of the existing transcripts
    This will:
    - Find the new end_site of the transcripts --> actual elongation
    - Determine the splicing status of the transcripts after elongation. Note: if a transcript is spliced at intron_i at previous time point, then it should also be spliced at intron_i at this time point
    - Determine the cleavage status of the transcripts after elongation. Note: if a transcript is cleaved at previous time point, then it should also be cleaved at this time point
    :param time_idx: current time idx
    :param time_since_prev: time (minutes) since previous time point (at index time_idx-1)
    :param trans_df: df that keeps track of all the transcripts at all the time points. Columns: time_idx, end_site, trans_idx, cleaved
    :param trans_spl_df: df that keeps track of the transcripts' splicing status
    :param gtf_df: gene annotation df, listing all the features of this gene
    :return: updated trans_df with transcripts of this particular
    """
    assert time_idx > 0, 'time_idx must be greater than 0 because we do not yet simulate transcript elongation at time_idx=0'
    # now, given the end point of transcripts in previous time point, we would like to calculate the endpoint of transcripts at this time point
    curr_stop = prev_trans_df.apply(lambda x: calculate_enlongated_endsite(prev_stop=x['end_site'], gtf_df=gtf_df, time_since_prev=time_since_prev), axis=1)  # calculate the stop site of each transcript at this time point
    curr_trans_df = pd.DataFrame({'time_idx': time_idx,
                                  'end_site': curr_stop.astype(int),
                                  'trans_idx': prev_trans_df['trans_idx']})  # df denoting the elongated transcript of the current time point
    # SPLICING of transcripts after elongation
    curr_trans_spl_df = simulate_splicing_from_curr_trans(curr_trans_df, gtf_df)  # df denoting the splicing status of the elongated transcript of the current time point after elongation. Columns: feature, Udist, Ilen, intron_h, txrate, transcript, end_site, splice_prob, is_spliced
    # next, we need a layer of sanity check: If in previous timepoint, a transcript is spliced at intron_i, then in this time point, it should also be spliced at intron_i
    prev_trans_spl_df = prev_trans_spl_df[['feature', 'trans_idx', 'end_site', 'splice_prob', 'is_spliced']]  # only keep the columns that we care about from the previous time point
    curr_trans_spl_df = curr_trans_spl_df.merge(prev_trans_spl_df, on=['feature', 'trans_idx'],
                                                suffixes=('_curr', '_prev'))  # columns: feature, Udist, Ilen, intron_h, txrate, transcript, end_site, splice_prob_curr, is_spliced_curr, end_site_prev, splice_prob_prev, is_spliced_prev
    curr_trans_spl_df['is_spliced_curr'] = curr_trans_spl_df.apply(lambda x: x['is_spliced_curr'] | x['is_spliced_prev'], axis=1)  # if this transcript is spliced at intron_i at previous time point, then it should also be spliced at intron_i at this time point
    curr_trans_spl_df.drop(['end_site_prev', 'splice_prob_prev', 'is_spliced_prev'], axis=1, inplace=True)  # drop the columns about previous time point's splicing status
    curr_trans_spl_df.rename(columns={'splice_prob_curr': 'splice_prob', 'is_spliced_curr': 'is_spliced', 'end_site_curr': 'end_site'}, inplace=True)  # rename the columns to be consistent with the simulation framework
    # CLEAVAGE of transcripts after elongation
    curr_trans_df = simulate_cleavage_from_curr_trans(curr_trans_df, gtf_df) # added column 'cleaved'
    # a layer of sanity check: if a transcript is cleaved at previous time point, then it should also be cleaved at this time point
    curr_trans_df['cleaved'] = curr_trans_df['cleaved'] | prev_trans_df['cleaved'] # if a transcript is cleaved at previous time point, then it should also be cleaved at this time point
    return curr_trans_df, curr_trans_spl_df

def create_new_transcript(time_since_prev, time_idx, gtf_df, init_trans):
    """
    This function will simulate new transcripts that get created since the previous time point and this time point
    :param time_since_prev: time (minutes) since previous time point (at index time_idx-1)
    :param time_idx: current time idx that we will observe the new transcripts
    :param gtf_df: gtf file, listing all the features of this gene
    :param init_trans: number of transcripts that are created during the time period between the previous time point and this time point
    :return: df denoting the newly created transcripts. columns: trans_idx, end_site, time_idx
    """
    # first, calculate the furthest possible end site of the transcript that can be created during this time period
    time_since_prev_feat = time_since_prev
    curr_feat_idx = 0
    while time_since_prev_feat > 0 and curr_feat_idx < gtf_df.shape[0]:
        time_since_prev_feat -= gtf_df.loc[curr_feat_idx, 'time']
        curr_feat_idx += 1
    # up until here, either time_since_prev_feat <= 0 or curr_feat_idx == gtf_df.shape[0]
    if time_since_prev_feat <= 0:
        furthest_end = gtf_df.loc[curr_feat_idx, 'end'] - (time_since_prev_feat * gtf_df.loc[curr_feat_idx, 'txrate'] * helper.ONE_KB)
    elif curr_feat_idx == gtf_df.shape[0]:  # if time_since_prev_feat > 0 and curr_feat_idx == gtf_df.shape[0] --> we reached the end of gene but we do not run out of time yet, which means that this transcript can be extended beyond the end of the gene
        furthest_end = gtf_df.loc[curr_feat_idx-1, 'end'] + (time_since_prev_feat * gtf_df.loc[curr_feat_idx-1, 'txrate'] * helper.ONE_KB)  # the furthest that it can go it at the end the of gene and imaginarily beyond the end of the gene by the same rate as the last feature
    else:
        assert False, 'This should not happen that we cannot determine the furthest end site of the transcript that can be created during this time period'
    # second, randomly sample the end site of the transcript for the newly created transcripts
    gene_start = gtf_df['start'].iloc[0]  # the start of the gene
    end_sites = np.random.randint(gene_start, furthest_end, init_trans)
    # third, return the df denoting the newly created transcripts
    df = pd.DataFrame({'time_idx': time_idx, 'end_site': end_sites, 'trans_idx': range(len(end_sites))})
    return df

def simulate_transcripts_new_timepoint(time_idx, time_since_prev, prev_trans_df, prev_trans_spl_df, gtf_df, new_rate, degrade_rate):
    """
    this function will create the data of transcripts in the new timepoint given the data from the previous time point. It includes:
    - elongate transcripts from previous time point --> checking splicing and cleavage status of the elongated transcripts
    - create new transcripts --> checking splicing and cleavage status of the newly created transcripts
    - degrade transcripts
    :param prev_trans_df: Data of the transcripts from previous time points.
    :param prev_trans_spl_df: data of the splicing status of transcripts from previous time point
    :param gtf_df: data of gene annotation features
    :param time_since_prev: time (minutes) since previous time point
    :param time_idx: current time idx
    :param new_rate: rate of new transcripts being initiated, reads/min
    :param degrade_rate: rate of transcripts being degraded, in proportion of current number of transcripts
    :return:
    """
    prev_num_trans = prev_trans_df.shape[0]  # number of transcripts at the previous time point
    num_trans_to_create = int(time_since_prev* new_rate)  # number of transcripts to create at this time point
    num_trans_to_drop = int(prev_num_trans * degrade_rate)  # number of transcripts to drop at this time point
    # elongate transcripts from previous time point
    elong_trans_df, elong_trans_spl_df = simulate_transcript_elongation(time_idx, time_since_prev, prev_trans_df, prev_trans_spl_df, gtf_df) # elongation of transcripts that were created before time t=0, then check the elongated transcripts' SPLICING and CLEAVAGE status.
    # CREATE NEW TRANSCRIPT --> then refer their splicing and cleavage status
    new_trans_df = create_new_transcript(time_since_prev, time_idx, gtf_df, num_trans_to_create)  # a df of transcripts. columns: trans_idx, end_site, time_idx
    new_trans_df['trans_idx'] = prev_trans_df['trans_idx'].iloc[-1] + 1 + new_trans_df['trans_idx']  # change the transcript idx so that we know these are new transcripts, created after the transcripts from the previous time point
    new_trans_spl_df = simulate_splicing_from_curr_trans(new_trans_df, gtf_df)  # splicing status of newly created transcripts
    new_trans_df = simulate_cleavage_from_curr_trans(new_trans_df, gtf_df)  # added column 'cleaved' --> cleavage status of newly created transcripts
    # DEGRADE TRANSCRIPTS
    drop_trans_idx = np.random.choice(elong_trans_df['trans_idx'], num_trans_to_drop, replace=False)
    # drop the transcripts that are degraded between previous and current time points --> both transripts and their splicing status should be updated
    trans_df = elong_trans_df[~elong_trans_df['trans_idx'].isin(drop_trans_idx)]
    trans_spl_df = elong_trans_spl_df[~elong_trans_spl_df['trans_idx'].isin(drop_trans_idx)]
    trans_df = pd.concat([trans_df, new_trans_df], ignore_index=True)  # add the newly initiated transcripts in the dataframe
    trans_spl_df = pd.concat([trans_spl_df, new_trans_spl_df], ignore_index=True)  # add the newly initiated transcripts in the dataframe
    return trans_df, trans_spl_df




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gtf_fn', type=str, required=True, help='gene annotation file, outlining all the introns and exons etc. of the gene')
    parser.add_argument('--label_time', default=[0, 5, 10], type=int, nargs='+', required=False, help='How many minutes that the mRNA molecule is exposed to the labeling reagent')
    parser.add_argument('--transcription_rate', default=1500, required=False, type=float, help='Transcription rate in nt/min, right now assumed constact across the gene')
    parser.add_argument('--target_exp', default=5, type=int, required=False, help='expression level of the gene at time t=0, in TPM. This may be changed in later implementation of the simulation because I am not sure if it is realisitic to assume that we know gene expression at t=0 since we are dealing with nascent mRNA. ')
    parser.add_argument('--num_total_transcript_millions', default=100, required=False, type=float, help='number of millions of transcripts that this cell/cells generated through transcription')
    parser.add_argument('--new_rate', default=5, required=False, type=int, help='Number of transcripts being initiated every minute')  #this is a simplifying assumption, because in reality, the rate of new transcripts being initiated is not constant, but rather depends on different biological factors
    parser.add_argument('--degrade_rate', default=0.03, required=False, type=float, help='rate of transcripts being degraded, in proportion of current number of transcripts')  #this is a simplifying assumption, because in reality, the rate of new transcripts being degraded is not constant, but rather depends on different biological factors such as presence of proteins protecing the mRNA from degradation, etc. The presence of proteins depend on the DNA sequence itself sometimes.
    # the reaason why we need the total number of transcript is because:
    # - TMP_i = (RPK_i / \sum_j-1^{num_gene} RPK_j) * 10^6--> relative abundance of transcription in gene i
    # --> TPM_i/10^6 = relative abundance of transcription in gene i compared to other genes
    # - RPK_i = number of reads mapped to gene i/ length of gene i --> control for hte fact that there are more reads in a longer gene than a shorter gene
    # If we want to simulate transcripts of a gene, given its TPM, we need to know how many transcript in total were generated from all the genes, this is what num_total_transcript_millions is for.
    # N_i = TPM_i/10^6 * num_total_transcript = TPM_i * num_total_transcript_millions--> number of transcripts of gene i
    parser.add_argument('--5eU_prob', type=float, default=0.1, help='probability of conversion for 5eU', required=False)
    parser.add_argument('--6sG_prob', type=float, default=0.1, help='probability of conversion for 6sG', required=False)
    parser.add_argument('--4sU_prob', type=float, default=0.1, help='probability of conversion for 4sU', required=False)
    parser.add_argument('--intron_h', default=5, required=False, type=int, help='half life of P(the mRNA molecule is spliced) in minutes. this is only needed if the gtf file does not already have half-life for each intron. If that is the case (no halflife in gtf file) then we will assign this rate to all the introns in the gene') # this is the half life of the splicing event, not the half life of the mRNA molecule
    parser.add_argument('--PAS_h', default=5, required=False, type=int,
                        help='half life of P(the mRNA is cleaved) in minutes. Right now, we are assuming that there is only one PAS on a gene, which is an oversimplication.')
    parser.add_argument('--RTR', default=100, required=False, type=int, help ='default readthrough region in bp, in case the gtf file does not have this information')
    parser.add_argument('--output_folder', type=str, required=True, help='output folder')
    args = parser.parse_args()
    helper.check_file_exist(args.gtf_fn)
    helper.make_dir(args.output_folder)
    assert 0 in args.label_time, 'label_time must contain 0 which is the start state of the experiment'
    print('Done getting input arguments')
    """
    Simulation framework: 
        At each time point: 
    -	Calculate the end point of each transcript. 
    -	Calculate the splicing probability of each intron on the transcript  sample from that. 
    -	Create mature transcript (spliced out or cleavaged or none)
    -	Introduce the 4sU, 5sE, etc. based on the replacement rates
    -	Create fragments
    -	Create reads from each fragments
    -	Create sequencing error based on error rates. 
    What are the factors that are being simplified by this simulation? 
    -	Transcripts are being continually created (we are ignoring the initiation rates right now)
    -	There are alternative splicing events (if we want to model this, we would give each exon the probability of being included or not?)
    -	Across one region intron/exon, the  transcription elongation rate is constant
    At each time point: 
    -	Calculate the end point of each transcript. 
    -	Calculate the splicing probability of each intron on the transcript  sample from that. 
    -	Create mature transcript (spliced out or cleavaged or none)
    -	Introduce the 4sU, 5sE, etc. based on the replacement rates: change bases based on the tags presence (which also depends on 
    -	Create fragments
    -	Create reads from each fragments
    -	Create sequencing error based on error rates. 
    What are the factors that are being simplified by this simulation? 
    -	Transcripts are being continually created (we are ignoring the initiation rates right now)
    -	There are alternative splicing events (if we want to model this, we would give each exon the probability of being included or not?)
    -	Across one region intron/exon, the  transcription elongation rate is constant
    """
    gtf_df = read_gtf_file(args.gtf_fn, args.intron_h, args.PAS_h, args.RTR)
    # simulate transcripts at time 0, with reads that have random end points
    trans_df_t0, trans_spl_df_t0 = simulate_starting_transcripts(gtf_df=gtf_df, init_exp=args.target_exp, num_total_transcript_millions=args.num_total_transcript_millions) # start transcripts with random end points, splicing and cleavage status
    # simulate the elongation of transcripts
    trans_df_t1, trans_spl_df_t1 = simulate_transcripts_new_timepoint(time_idx=1, time_since_prev=args.label_time[1]-args.label_time[0], prev_trans_df=trans_df_t0, prev_trans_spl_df=trans_spl_df_t0, gtf_df=gtf_df, new_rate=args.new_rate, degrade_rate=args.degrade_rate)
    trans_df_t2, trans_spl_df_t2 = simulate_transcripts_new_timepoint(time_idx=2, time_since_prev=args.label_time[2]-args.label_time[1], prev_trans_df=trans_df_t1, prev_trans_spl_df=trans_spl_df_t1, gtf_df=gtf_df, new_rate=args.new_rate, degrade_rate=args.degrade_rate)  # simulate the elongation of transcripts from t1 to t2
    # simulate the conversion of 5eU, 6sG, 4sU
