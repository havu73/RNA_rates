import numpy as np
from numpy.testing import assert_array_equal
import pandas as pd
from . import helper
from .elongation_calculation import calculate_startsite_given_pred_h, calculate_enlongated_endsite, time_to_elongate
from .simulate_reads_from_transcripts import calculate_breakpoints_weilbull_fragment, check_intron_break_per_read, adjust_absolute_one_read_coords, calculate_breakpoints_uniform_fragment, calculate_breakpoints_with_edge_effects, get_pairedEnd_reads_df
OVERFLOW_LOG2_LOWLIMIT = -1000  # if x <OVERFLOW_LIMIT then we can assume that 2**x = 2**OVERFLOW_LIMIT

def calculate_splice_prob(endpoint, row, elongf_df=None):
    """
    Given one transcript's end point and the row's data about the intron's splicing parameters, we want to calculate the probability that this transcript is spliced at this particular intron
    :param endpoint: the ABSOLUTE coord of transcript's endpoint
    :param row: the row of the intron in the splice_df
    :return: the probability that this transcript is spliced at this intron
    """
    if row['intron_h'] == np.inf:
        return 0
    time_since_endI = time_to_elongate(prev_stop= row['end'], curr_stop=endpoint, elongf_df=elongf_df, e_colname='txrate')
    log_UNspice_prob = -(time_since_endI) / (row['intron_h']) # log(unsplice_prob) = 0 when endpoint <= row['start'] + row['Ilen'] (the end of the intron), and decreases to -inf as endpoint increases more downstream of the intron end. (prob. UNsplice decreases).
    log_UNspice_prob = 0 if log_UNspice_prob > 0 else log_UNspice_prob # if the logprob is negative (endsites before the gene end), set it to 0 --> cannot be spliced yet.
    log_UNspice_prob = OVERFLOW_LOG2_LOWLIMIT if log_UNspice_prob < OVERFLOW_LOG2_LOWLIMIT else log_UNspice_prob # if the logprob is too negative, set it to -1000 to avoid overflow
    splice_prob = 1 - 2 ** log_UNspice_prob  # the probability that this transcript is spliced at this intron
    # splice_prob = 0 when endpoint = row['start'] + row['Ilen'] (the end of the intron)
    # splice_prob = 1/2  when endpoint is equal to the expected distance when the transcript travels beyond the end of the intron in half-life h minutes
    # Note that here we calculate the distance of the end point (ABSOLUTE coord) compared to the end of the intron (ABSOLUTE coord), and the halflife of the intron is in minutes, and the transcription rate is in kb/min.
    return splice_prob


class Transcript:
    '''
    The unit of length in gtf_df and elongf_df should both be in bp.
    '''
    gtf_df = None # class variable, shared across all instances of the Transcript class, contains information about gene features
    elongf_df = None # class variable, shared across all instances of the Transcript class, contains information about elongation rates across the gene
    gene_start = None # class variable, shared across all instances of the Transcript class, contains information about the ABSOLUTE coordinate of the start of the gene
    # below are class parameters that are associated with the generations of transcripts' fragments and reads
    eta_val= helper.DFT_ETA_VALUE
    insertsize_min= helper.DFT_INSERTSIZE_MIN
    insertsize_max = helper.DFT_INSERTSIZE_MAX
    read_length= helper.DFT_READ_LENGTH

    @classmethod
    def set_gtf_df(cls, gtf_df: pd.DataFrame):
        """
        Set the gtf_df class variable.
        :param gtf_df: the gtf dataframe containing information about the gene. Required columns: feature, start, end, length, intron_h, PAS_h, time
        :return: None
        """
        cls.gtf_df = gtf_df.reset_index(drop=True)  # reset the index of the gtf_df to be 0-based
        cls.gtf_df.sort_values('start', inplace=True) # sort the features in the gene by their start site
        cls.gene_start = gtf_df['start'].iloc[0]
        cls.PAS_start_coord = (gtf_df[gtf_df['feature'] == 'PAS'])['start'].iloc[0]  # the genomic coordinate of the PAS start site (the end of the gene). It is possible that a gene has multiple PAS, and assume that features in gene are sorted in ascending order of their start site. Here, we find the first PAS in the gene, which is usually the constitutuve PAS. This function right now can only simulate the case of 1 PAS
        cls.PAS_end_coord = (gtf_df[gtf_df['feature'] == 'PAS'])['end'].iloc[0]  # the genomic coordinate of the PAS END site (the end of the gene). It is possible that a gene has multiple PAS, and assume that features in gene are sorted in ascending order of their start site. Here, we find the first PAS in the gene, which is usually the constitutuve PAS. This function right now can only simulate the case of 1 PAS
        cls.end_of_gene = gtf_df['end'].iloc[-1]  # the genomic coordinate of the end of the gene (this should be the end of the RTR)
        cls.PAS_h = (gtf_df[gtf_df['feature'] == 'PAS'])['PAS_h'].iloc[0]  # cleavage half life, the first PAS in the gene
        cls.num_introns = gtf_df[gtf_df['is_intron']].shape[0]  # number of introns in the gene

    @classmethod
    def set_elongf_df(cls, elongf_df: pd.DataFrame=None):
        """
        Set the elongf_df class variable.
        :param elongf_df: the elongation dataframe containing information about the elongation rates across the gene. Required columns: txrate, start, end
        :return: None
        """
        if elongf_df is None:  # by default, we allow the elongf_df to be the same as the gtf_df that specify feature-specific elongation rates
            cls.elongf_df = cls.gtf_df.copy()
            assert 'txrate' in cls.elongf_df.columns, "elongf_df must contain a column 'txrate'"
            cls.set_PAS_txrate()
            return
        cls.elongf_df = elongf_df.reset_index(drop=True)  # reset the index of the elongf_df to be 0-based
        cls.elongf_df.sort_values('start', inplace=True) # sort the features in the gene by their start site
        required_colunmns = {'txrate', 'start', 'end'}
        if not required_colunmns.issubset(cls.elongf_df.columns):
            raise ValueError(f"elongf_df must contain columns: {required_colunmns}")
        # check that the last elongation rate's endpoint is np.inf
        if cls.elongf_df['end'].iloc[-1] != np.inf:
            raise ValueError("The last elongation rate's endpoint must be np.inf")
        cls.set_PAS_txrate()
        return

    @classmethod
    def set_PAS_txrate(cls):
        """
        Set the PAS_txrate class variable.
        :return: None
        """
        # find the row within elongf_df that corresponds to the PAS feature
        pas_idx = cls.elongf_df[(cls.elongf_df['start'] <= cls.PAS_start_coord) & (cls.elongf_df['end'] >= cls.PAS_end_coord)].index[0]
        cls.PAS_txrate = cls.elongf_df.loc[pas_idx]['txrate']
        return

    @classmethod
    def set_read_params(cls, pair_end= False, eta_val: int= helper.DFT_ETA_VALUE, insertsize_min: int = helper.DFT_INSERTSIZE_MIN, insertsize_max: int = helper.DFT_INSERTSIZE_MAX, read_length: int= helper.DFT_READ_LENGTH, frag_func: str = 'weibull', unif_avg_frag_len: int=250):
        """
        Set the parameters that are essential to simulating fragments and transcript. These class variables--> shared across instances of the Transcript class.
        :param eta_val: eta value input to the Weibull distrubtion
        :param insertsize_min: minimum length of selected fragment (size-select)
        :param insertsize_max: maximum length of selected fragment (size-select)
        :return: None
        """
        assert frag_func in helper.FRAGMENT_DIST_LIST, "frag_func must be either in: "+str(helper.FRAGMENT_DIST_LIST)
        if frag_func == 'weibull':
            cls.frag_func = calculate_breakpoints_weilbull_fragment
        elif frag_func == 'uniform':
            cls.frag_func = calculate_breakpoints_uniform_fragment
        elif frag_func == 'weibull_edge':
            cls.frag_func = calculate_breakpoints_with_edge_effects
        cls.eta_val = eta_val
        cls.insertsize_min = insertsize_min
        cls.insertsize_max = insertsize_max
        cls.read_length = read_length
        cls.unif_avg_frag_len = unif_avg_frag_len
        cls.pair_end = pair_end
        if pair_end: # make sure that insertsize_min is at least 2*read_length
            assert cls.insertsize_max != -1, "insertsize_max must be specified when pair_end is True"
            assert cls.insertsize_max != -1, 'Insertsize_max must be specified when pair_end is True'
            assert cls.read_length != -1, 'read_length must be specified when pair_end is True'
            assert cls.insertsize_min >= 2*cls.read_length, "insertsize_min must be at least 2*read_length when pair_end is True"



    def __init__(self, trans_idx: int, endpoint: int, splice_df:pd.DataFrame = None, reads_df: pd.DataFrame = None, set_degrade: bool = False, set_cleaved: bool = False, simulate_cleavage: bool = True):
        """
        Initialize the Transcript object.
        :param trans_idx: index of the transcript.
        :param endpoint: Integer representing the end point of the transcript. This end point is relative to the start of the gene, which is a class characteristic
        gene_start and other gene features should be defined in gtf_df that the users specify
        :param splice_df: a dataframe outlining the splicing patterns at different introns. Default to an empty dataframe.
        :param reads_df: a dataframe outlining the reads that stem from this transcript. Deafult to an empty dataframe.
        :param set_degrade: if True, then set the transcript to be degraded regardless of its probability of being degraded (this happens when a transcript is alongated from a previously-degraded transcript in a previous time point). Default to False.
        :param set_cleaved: if True, then set the transcript to be cleaved regardless of its probability of being cleaved (this happens when a transcript is alongated from a previously-cleaved transcript in a previous time point). Default to False.
        :param simulate_cleavage: if True, then simulate the cleavage event. Default to True. If not (in some cases, I do want that), then the transcript will not be cleaved regardless of its probability of being cleaved.
        """
        self.trans_idx = trans_idx
        self.endpoint = endpoint # ABSOLUTE coordinate, which means the first bp of the chromosome is 0
        self.splice_df = splice_df
        self.is_cleaved = False
        self.is_mature = False
        self.is_degrade = False
        self.determine_splicing()  # as soon as we initiate a transcript, we should also calculate its splicing patterns, but we do not need to calculate the reads yet so self.reads_df can be empty
        self.reads_df = reads_df # it can be None (not created yet), it can be a dataframe (empty or not empty)
        self.degrade(set_degrade) # If we want to set the transcript to be degraded regardless of its probability of being degraded (this happens when a transcript is alongated from a previously-degraded transcript in a previous time point). Default to False.
        self.simulate_cleavage = simulate_cleavage  # we do not set simulate_cleavage as a class variable because we want Experiment to have control over how we want to simulate the transcripts.
        self.determine_cleavage(set_cleaved, self.simulate_cleavage) # If we want to set the transcript to be cleaved regardless of its probability of being cleaved (this happens when a transcript is alongated from a previously-cleaved transcript in a previous time point). Default to False.
        # DO NOT CALL self.evaluate_transcript_mature() here. It is a very special functionality that should only be called on in the Experiment object, and not here.
        self._set_length()

    def determine_splicing(self):
        """
        Determine the splicing patterns of the transcript.
        :param gtf_df: the gtf dataframe containing information about the gene. Required columns: feature, start, end, length, intron_h, PAS_h, time
        :return: modified self.splice_df to include the splicing patterns of the transcript.
        """
        if self.splice_df is None:
            self._create_splice_df()
        elif self.splice_df.empty:
            self._create_splice_df()
        else:
            self._validate_splice_df()

    def _create_splice_df(self):
        """
        Create the splice dataframe, which contain information about each of the introns in the gene, and their splicing status within the transcript.
        :return: create self.splice_df. Columns: 'start', 'Ilen', 'intron_h'
        """
        self.splice_df = Transcript.gtf_df[Transcript.gtf_df['is_intron']==True][['start', 'end', 'length', 'intron_h']].copy()
        self.splice_df.sort_values(['start'], inplace=True)
        self.splice_df.reset_index(inplace=True, drop=True) # Now: index of the splice_df is the index of the intron in the gene (0-based)
        self.splice_df.rename(columns={'length': 'Ilen'}, inplace=True)  # rename the columns to be consistent with the simulation framework
        self.splice_df['splice_prob']= self.splice_df.apply(lambda row: calculate_splice_prob(self.endpoint, row, elongf_df=Transcript.elongf_df), axis=1)
        self.splice_df['splice_prob'] = self.splice_df['splice_prob'].apply(lambda splice_prob: 0 if splice_prob<0 else splice_prob)
        # the probability that this transcript is spliced at this intron P(spliced|endpoint=0, intron_h, txrate) = 0 and P(spliced|endpoint=intron_h*txrate, intron_h, txrate) = 1/2, where endpoint is relative to the END of the intron
        # so P(spliced| endpoint) = 1- 2**{-\frac{x}{hr}}, where x is endsite RELATIVE TO INTRON END, and h is the half life of splicing in minutes and r is transcription rate in bp (not KB)/min
        # sample whether each intron has been spliced or not on this transcript given the splice probability
        self.splice_df['is_spliced'] = np.random.binomial(1, self.splice_df['splice_prob'])
        self._calculate_splice_stats()  # calculate a few more statistics about the splice sites that will come in handy later when we map reads to the gene
        return

    def _calculate_splice_stats(self):
        """
        Given that the splicing status of introns are determined within this transcript, we want to calculate a few more statistics about the splicing patterns of the transcript. this function is broken down from the _create_splice_df function because there are contexts where it is used without recalculating the splicing status of the introns.
        :return: modified self.splice_df to include columns: spliced_out_len, Blen, start_within_trans
        """
        # now calculate a few more statistics about the splice sites that will come in handy later when we map reads to the gene
        # Calculate Blen: the length of total spliced out region by the beginning of any reads that start right before this intron starts (but also after the previous intron ends)
        self.splice_df['spliced_out_len'] = self.splice_df['Ilen'] * self.splice_df['is_spliced']
        self.splice_df['Blen'] = self.splice_df['spliced_out_len'].transform(lambda x: x.cumsum().shift(fill_value=0))  # calculate the cumulative length of the introns that are spliced out up until the BEGINING of the current intron (in the current row) of each transcript.
        self.splice_df['start_within_trans'] = self.splice_df['start'] - Transcript.gene_start - self.splice_df['Blen']  # calculate the start of each intron within the transcript length
        return

    def _validate_splice_df(self):
        """Validate the splice dataframe structure."""
        num_introns = Transcript.gtf_df[Transcript.gtf_df['is_intron']].shape[0]  #number of introns in the gene
        assert self.splice_df.shape[0] == num_introns, f"splice_df must have {num_introns} rows, but has {self.splice_df.shape[0]} rows instead."
        assert_array_equal(self.splice_df['start'].values, Transcript.gtf_df[Transcript.gtf_df['is_intron']]['start'].values)
        assert_array_equal(self.splice_df['Ilen'].values, Transcript.gtf_df[Transcript.gtf_df['is_intron']]['length'].values)
        self._calculate_splice_stats() # this is needed because when we inherit splice_df from transcript of a previous time point, we do not calculate the splice stats yet. Calculate here beacuse these stats are needed for the read mapping step
        expected_columns = {'splice_prob', 'is_spliced', 'Ilen', 'Blen', 'start_within_trans', 'spliced_out_len'}
        if not expected_columns.issubset(self.splice_df.columns):
            raise ValueError(f"splice_df must contain columns: {expected_columns}")

    def determine_cleavage(self, set_cleaved: bool = False, simulate_cleavage: bool = True):
        """
        Determine the cleavage patterns of the transcript.
        :param set_cleaved: if True, then set the transcript to be cleaved regardless of its probability of being cleaved (this happens when a transcript is alongated from a previously-cleaved transcript in a previous time point). Default to False.
        :param simulate_cleavage: if True, then simulate the cleavage event. Default to True. If not (in some cases, I do want that), then the transcript will not be cleaved regardless of its probability of being cleaved.
        :return: add attributes self.is_cleaved (boolean), and self.cleaved_prob (integer) to the transcript.
        And then modify the end point of the transcript to be the cleavage site.
        """
        if simulate_cleavage == False:  # if we do not want to simulate the cleavage event, then the transcript will not be cleaved regardless of its probability of being cleaved.
            self.is_cleaved = False
            self.cleaved_prob = 0
            return
        if set_cleaved: # if we want to set the transcript to be cleaved regardless of its probability of being cleaved
            # this happens only when a transcript is alongated from a previously-cleaved transcript in a previous time point
            self.is_cleaved = set_cleaved
            self.cleaved_prob = 1
            self.endpoint = Transcript.PAS_end_coord
            return
        log_UNcleaved_prob = -(self.endpoint - Transcript.PAS_end_coord) / (Transcript.PAS_h * Transcript.PAS_txrate * helper.ONE_KB)  # log(uncleaved_prob) = 0 when endpoint <= PAS_end_coord (the end of the gene), and decreases to -inf as endpoint increases more downstream of the PAS end site. (prob. UNcleave decreases).
        log_UNcleaved_prob = 0 if log_UNcleaved_prob > 0 else log_UNcleaved_prob  # if the logprob is positive (endsites before the gene end), set it to 0 --> cannot be cleaved yet.
        log_UNcleaved_prob = OVERFLOW_LOG2_LOWLIMIT if log_UNcleaved_prob < OVERFLOW_LOG2_LOWLIMIT else log_UNcleaved_prob  # if the logprob is too negative, set it to -1000 to avoid overflow
        self.cleaved_prob = 1 - 2 ** log_UNcleaved_prob
        # for each transcript, the probability that has been cleaved given its end site and the cleavage half life
        self.is_cleaved = (np.random.binomial(1, self.cleaved_prob) == 1)
        self.endpoint = Transcript.PAS_end_coord * self.is_cleaved + (1-self.is_cleaved) * self.endpoint
        # if the transcript is cleaved, then the end point is the PAS end site, otherwise, the end point is the end point of the transcript
        return

    def is_intron_spliced(self, intron_index):
        """
        Given the index of the intron, this function will return True/False based on whether this intron is spliced or not in this transcript.
        :param intron_index: index of intron, associated with the row index of splice_df
        :return: True (spliced) or False (unspliced)
        """
        return self.splice_df.loc[intron_index, 'is_spliced'] == 1


    def elongate(self, time):
        """
        Elongate the transcript by time minutes.
        :param time: # minutes to elongate the transcript to
        :return: a new end point, new splice_df based off of the current splice_df (if an intron is already spliced, it will continue to be spliced)
        """
        if self.is_degrade or self.is_mature:  # if degrade, we return the existing values of the object, and handling of degraded transcripts will be done in Experiment object
            return self.endpoint, self.splice_df, self.is_cleaved, self.is_degrade
        if self.is_cleaved:  # endpoint is previously set to PAS_end_coord, so we keep the same endpoint
            return self.endpoint, self.splice_df, self.is_cleaved, self.is_degrade
        next_stop = calculate_enlongated_endsite(self.endpoint, Transcript.elongf_df, time)
        next_splice_df = self.splice_df[['start', 'end', 'Ilen', 'intron_h', 'is_spliced']].copy()
        next_splice_df.rename(columns={'is_spliced': 'is_spliced_prev'}, inplace=True)
        next_splice_df['splice_prob'] = next_splice_df.apply(lambda row: calculate_splice_prob(next_stop, row, elongf_df=Transcript.elongf_df), axis=1)
        next_splice_df['splice_prob'] = next_splice_df.apply(lambda row: 1 if row['is_spliced_prev'] else row['splice_prob'], axis=1)  # if this intron is previously spliced, then it definitely stays spliced at this time point
        next_splice_df['splice_prob'] = next_splice_df['splice_prob'].apply(lambda splice_prob: 0 if splice_prob < 0 else splice_prob)
        next_splice_df['is_spliced'] = np.random.binomial(1, next_splice_df['splice_prob'])
        next_splice_df.drop(['is_spliced_prev'], axis=1, inplace=True)
        return next_stop, next_splice_df, self.is_cleaved, self.is_degrade

    def evaluate_transcript_mature(self):
        """
        Here, we set a very simple rule to determine if a transcript is mature or not.
        A transcript is mature if it is cleaved and spliced at all introns.
        :return:
        """
        self.is_mature= False
        if (not self.simulate_cleavage) and (self.endpoint > Transcript.end_of_gene):  # if we do not want to simulate the cleavage event, then the transcript will not be cleaved regardless of its probability of being cleaved.
            self.is_mature = True
        if self.simulate_cleavage: # if we do simulate the cleavage event, then a transcript is mature when it is cleaved and spliced at all introns
            if self.is_cleaved and self.splice_df['is_spliced'].all():
                self.is_mature = True
            else:
                self.is_mature = False
        if self.is_mature:
            self.degrade(set_degrade=False)
        return

    def degrade(self, set_degrade: bool = True):
        """
        Degrade the transcript. this will just add a flag of degrade= True and delete the reads_df and splice_df of this transcript (to save space). The reason why this function is needed is because if we simply delete the transcript object, it will be later hard to keep track of what transcripts got degraded.
        :param set_degrade: if True, then set the transcript to be degraded regardless (this happens when a transcript is alongated from a previously-degraded transcript in a previous time point). Default to True.
        :return: add attribute self.degrade (boolean) to the transcript. delete the large dataframes self.reads_df and self.splice_df
        """
        self.is_degrade = set_degrade
        if self.is_degrade:
            del(self.splice_df)
            del(self.reads_df)
            self.splice_df = pd.DataFrame()
            self.reads_df = pd.DataFrame()  # after deleting the dataframes, we want to replace them with empty to avoid errors when we reference them later
            self.is_degrade=True
        return

    def _set_length(self):
        """
        Calculate the length of the transcript. This function should only be called after we have defined the splice_df and set the shared values of the class variable (gtf_df, gene_start).
        :return: length of the transcript.
        """
        if self.is_mature or self.is_degrade:  # if the transcript is mature or degraded, then the length is 0
            # I checked that a transcript is never simulated to become mature when I use PBD experiment.
            # The only time that we actually evaluate a transcript' maturity is initiaion of experiments without PBD.
            self.length = 0
            return
        if Transcript.gene_start != None and not self.splice_df.empty:
            self.length = self.endpoint - Transcript.gene_start - (self.splice_df['Ilen']*self.splice_df['is_spliced']).sum()
        elif Transcript.gene_start == None:
            raise ValueError("gene_start must be defined in the gtf_df class variable")
        elif self.splice_df.empty and Transcript.num_introns != 0:
            raise ValueError("splice_df must be defined before calculating the length of the transcript")
        elif self.splice_df.empty and Transcript.num_introns == 0:
            self.length = self.endpoint - Transcript.gene_start
        return

    @property
    def get_length(self):
        """
        Get the length of the transcript.
        :return:
        """
        if self.length == None:
            self._set_length()
        return self.length

    def determine_reads(self):
        """
        Determine the reads that stem from this transcript.
        :param eta_val: eta value input to the Weibull distrubtion
        :param insertsize_min: minimum length of selected fragment (size-select)
        :param insertsize_max: maximum length of selected fragment (size-select)
        :return: modified self.reads_df to include the reads that stem from this transcript.
        """
        if self.reads_df is None:  # it has not been created yet
            self._create_reads_df()
        else:
            self._validate_reads_df() # this will make self.reads_df be a dataframe with the correct columns, even if it is empty (no reads beacuse of short tra

    def _create_reads_df(self):
        """
        Create the reads dataframe, which contain information about each of the reads.
        :param eta_val: eta value input to the Weibull distrubtion, used to generate the length of the fragments
        :param insertsize_min: minimum length of selected fragment (size-select)
        :param insertsize_max: maximum length of selected fragment (size-select)
        :param read_length:
        :return:
        """
        if self.is_mature or self.is_degrade:  # if the transcript is mature or degraded, then the length is 0
            self.reads_df = pd.DataFrame()
            return
        # first, we will generate reads by breaking down the transcript into fragments, size-select fragments, then for each fragment, reads correspond to the first read_length bp of the fragment
        # this will generate the relative start and end of the reads with respect to the length of the transcript
        self._set_length()
        if self.length == 0:
           # the only time that a transcript has length 0 is actually happens when there is some new transcripts created but the wiggle room functionality forces these transcripts to have length 0
            self.reads_df = pd.DataFrame()
            return # if the length of the transcript is 0, then we do not need to generate any reads
        # n stands for the number of fragments that we will break the transcript into
        frag_length, frag_start = Transcript.frag_func(self.length, eta_val=Transcript.eta_val, avg_frag_len=Transcript.unif_avg_frag_len) # array frag_length: length of each fragment, array frag_start: start of each fragment
        self.reads_df = pd.DataFrame()
        self.reads_df['rel_start'] = frag_start
        self.reads_df['frag_len'] = frag_length
        self.reads_df['frag_idx'] = range(len(frag_length))  # the index of the fragment that the read comes from
        if Transcript.insertsize_min != -1 and Transcript.insertsize_max != -1:  # if either of them is -1, that means users do not want to generate reads but rather want to keep the whole transcripts, so we break the transcripts into fragments but we do not get rid of any part of any fragment
            self.reads_df = self.reads_df[(self.reads_df['frag_len'] >= Transcript.insertsize_min) & (self.reads_df['frag_len'] <= Transcript.insertsize_max)]  # size select the fragment
        else: # we accept all fragments, but we need to filter fragments with length 0
            self.reads_df = (self.reads_df[self.reads_df['frag_len'] > 0])
        # fourth, for each selected fragment, obtain the reads. Each fragment generate one read.
        if Transcript.read_length != -1:  # if users sepecify read_length as -1 that means they do not want to get rid of any portion of the trasncripts (we can break the transcripts into fragments, but we will not take the first read_length bp of each fragment as reads. Instead, we will keep the whole fragment as a read)
            self.reads_df['rel_end'] = self.reads_df['rel_start'] + Transcript.read_length  # calculate the end of the read
        else:
            self.reads_df['rel_end'] = self.reads_df['rel_start'] + self.reads_df['frag_len']
        self.reads_df['read_len'] = self.reads_df['rel_end'] - self.reads_df['rel_start']
        # now, we will check if we need to simulate pair-end reads. If yes, then we will generate the second read of the pair
        if Transcript.pair_end:
            self.reads_df = get_pairedEnd_reads_df(self.reads_df)  # this will add the second read of the pair to the reads_df, and the middle part so that with paired_end reads we get the whole fragment
        # second, we will map the reads to the length of the gene based on the relative start and end of the reads with transcript length. This step involves adjusting the reads based on the splicing patterns of the transcript
        # check if the reads_df is empty. there can be cases where a transcript does not produce any fragments that pass the length filter and hence have no reads --> therefore we wont need to map the reads to the gene
        if not self.reads_df.empty:
            self._map_reads_to_gene()
            self.reads_df['trans_idx'] = self.trans_idx


    def _map_reads_to_gene(self):
        """
        This function is used after we have found the relative start and end of the reads with respect to the length of the transcript. We will map the reads to the gene based on the splicing patterns of the transcript.
        :return: modified self.reads_df to include abs_start, abs_end, subtract_start, subtract_end, IE, EE, EI, PAS_overlap --> information about reads' alignment status given the splicing patterns of the reads
        """
        if not self.splice_df.empty:  # if the transcript has introns, do the following steps before calculating the absolute start and end coordinate of the reads
            # if the transcript has no introns, don't do it and just calculating the absolute start and end coordinate of the reads
            intron_read_start = [0] + list(self.splice_df['start_within_trans']) + [self.length]  # list recording the start of the introns within the transcript length. (plus the first and last entry which are 0 and transcript length)
            # get the splicing status of each intron in this transcript (plus the first and last intron which are not spliced out)
            is_spliced = [0] + list(self.splice_df['is_spliced']) + [0]
            # we will find the intron index that immediately precedes the read, and splicedO_b4_reads will be the spliced out length that immediately precedes the read.
            self.reads_df[['overlap_splicedI', 'precedeI']] = self.reads_df.apply(lambda row: check_intron_break_per_read(row, intron_read_start, is_spliced), axis=1) # for each read, this function will return two values: (1) whether the read overlaps with a spliced out intron, (2) the index of the spliced out intron BEFORE the read
            # now we will adjust the reads so that they have the absolute start and end coordinate
        self.reads_df[['abs_start', 'abs_end', 'subtract_start', 'subtract_end']] = self.reads_df.apply(lambda row: adjust_absolute_one_read_coords(row, self.splice_df, Transcript.gene_start), axis=1)   # adjust the absolute start and end coordinate of the reads
        return

    def _validate_reads_df(self):
        """Validate the reads dataframe structure."""
        expected_columns = ['rel_start', 'rel_end', 'abs_start', 'abs_end', 'subtract_start', 'subtract_end']
        if self.reads_df.empty:  # this case happens when the transcript does not produce any fragments that pass the length filter and hence have no reads
            self.reads_df = pd.DataFrame(columns = expected_columns)
        if not set(expected_columns).issubset(self.reads_df.columns):
            raise ValueError(f"reads_df must contain columns: {expected_columns}")


