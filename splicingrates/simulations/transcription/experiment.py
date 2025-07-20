import pandas as pd
import numpy as np
import os
import json # for saving class attributes
from . import helper
from .transcripts import Transcript, calculate_enlongated_endsite
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

def determinte_IE_junction(read_row, intron_df):
    # This function should be applied to each row in the reads_df --> determine whether or not the read is an IE or EI junction read
    # read_row: a row in the reads_df. Required columns: subtract_start, subtract_end, abs_start, abs_end
    # intron_df: dataframes showing the intron start and end positions. Required columns: feature, start, end. The rows are already sorted in ascending order
    # return: intron index ('intron_1' etc.) if the read is an EI junction read, otherwise return None
    # Note that we do not care about IE junction, because if we count both IE and EI junctions, it could correspond to double-counting the number of trasncripts
    # that contain this unspliced intron
    read_start = read_row['abs_start']
    read_end = read_row['abs_end']
    if read_row['subtract_start'] != read_row['subtract_end']:  # this is actually an exon-exon junction read
        return None
    for _, intron_row in intron_df.iterrows():
        intron_start = intron_row['start']
        intron_end = intron_row['end']
        if (intron_start <= read_start <= intron_end) and (read_end > intron_end):  # an intron exon junction read
            return intron_row['feature']
        # if (intron_start <= read_end <= intron_end) and (read_start < intron_start):  # and exon intron junction read
        #     return intron_row['feature']
    return None

class Experiment:
    '''
    The unit of length in gtf_df and elongf_df should both be in bp.
    '''
    gtf_df = None # class variable, shared across all instances of the Experiment class, contains information about gene features
    elongf_df = None # class variable, shared across all instances of the Experiment class, contains information about elongation rates of the bins along the gene
    gene_start = None # class variable, shared across all instances of the Experiment class, contains information about the ABSOLUTE coordinate of the start of the gene
    # below are class parameters that are associated with the generations of transcripts' fragments and reads
    eta_val= helper.DFT_ETA_VALUE
    insertsize_min= helper.DFT_INSERTSIZE_MIN
    insertsize_max = helper.DFT_INSERTSIZE_MAX
    read_length= helper.DFT_READ_LENGTH
    # below are class parameters that are associated with the generation of transcripts
    lambda_init = DFT_LAMBDA_INIT
    burst_size = DFT_BURST_SIZE
    wiggle_room = DFT_WIGGLE_ROOM
    @classmethod
    def set_gtf_df(cls, gtf_df: pd.DataFrame):
        """
        Set the gtf_df class variable.
        :param gtf_df: the gtf dataframe containing information about the gene. Required columns: feature, start, end, length, intron_h, PAS_h, time
        :return: None
        """
        cls.gtf_df = gtf_df.reset_index(drop=True)  # reset the index of the dataframe (0,1,2,3,...) and drop the old index (which is the time index of the experiment)
        cls.gene_start = gtf_df['start'].iloc[0]
        cls.PAS_coord = (gtf_df[gtf_df['feature'] == 'PAS'])['start'].iloc[0]  # the genomic coordinate of the PAS start site (the end of the gene). It is possible that a gene has multiple PAS, and assume that features in gene are sorted in ascending order of their start site. Here, we find the first PAS in the gene, which is usually the constitutuve PAS. This function right now can only simulate the case of 1 PAS
        cls.PAS_h = (gtf_df[gtf_df['feature'] == 'PAS'])['PAS_h'].iloc[0]  # cleavage half life, the first PAS in the gene
        cls.num_introns = len(gtf_df[gtf_df['is_intron']])  # number of introns in the gene
        return

    @classmethod
    def set_elongf_df(cls, elongf_df:pd.DataFrame=None):
        """
        Set the elongf_df class variable
        :param elongf_df: the elongation rate dataframe. Required columns: feature, txrate, start, end
        :return: None
        """
        if elongf_df is None:  # by default, we allow the elongf_df to be the same as the gtf_df that specify feature-specific elongation rates
            cls.elongf_df = cls.gtf_df.copy()
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
        pas_idx = cls.elongf_df[(cls.elongf_df['start'] <= cls.PAS_coord) & (cls.elongf_df['end'] >= (cls.PAS_coord+1))].index[0]
        cls.PAS_txrate = cls.elongf_df.loc[pas_idx]['txrate']
        return

    @classmethod
    def set_read_params(cls, pair_end: bool=False, eta_val: int= helper.DFT_ETA_VALUE, insertsize_min: int = helper.DFT_INSERTSIZE_MIN, insertsize_max: int = helper.DFT_INSERTSIZE_MAX, read_length: int= helper.DFT_READ_LENGTH):
        """
        Set the parameters that are essential to simulating fragments and transcript. These class variables--> shared across instances of the Transcript class.
        :param eta_val: eta value input to the Weibull distrubtion
        :param insertsize_min: minimum length of selected fragement (size-select)
        :param insertsize_max: maximum length of selected fragement (size-select)
        :return: None
        """
        cls.eta_val = eta_val
        cls.insertsize_min = insertsize_min
        cls.insertsize_max = insertsize_max
        cls.read_length = read_length
        cls.pair_end = pair_end

    @classmethod
    def set_burst_init_params(cls, lambda_init: float=DFT_LAMBDA_INIT, burst_size: int=DFT_BURST_SIZE, wiggle_room: float=DFT_WIGGLE_ROOM):
        """
        Set the parameters that are essential to simulating fragments and transcript. These class variables--> shared across instances of the Transcript class.
        :param lambda_init: the rate of transcription initiation bursts (lambda_init events per minute)
        :param burst_size: the number of transcripts that are generated in each burst
        :param wiggle_room: If we are at time t where burst just happened. Next, we sample the time interval between the last burst (at t) and the next burst (exponential with scale 1/lambda_init), then the trnascripts are created as part of the next burst will be scattered around time [t +- wiggle_room * time_interval]. This is more reasonable because we do not want to have all the transcripts created at the EXACTLY the same time (burst moment).
        :return: None
        """
        cls.lambda_init = lambda_init
        cls.burst_size = burst_size
        cls.wiggle_room = wiggle_room

    @classmethod
    def set_simulate_cleavage(cls, simulate_cleavage: bool=True):
        """
        Set the simulate_cleavage class variable
        :param simulate_cleave: boolean, whether or not we want to simulate the cleavage of the transcripts in this experiment.
        :return: None
        """
        cls.simulate_cleavage = simulate_cleavage

    @classmethod
    def save_class_variables(cls, folder):
        """
        Save the class variables into a file
        :param folder: the folder that the file will be saved to
        :return: None
        """
        helper.make_dir(folder)
        # first, save the simple attributes to json
        class_simple_atts = ['simulate_cleavage', 'gene_start', 'PAS_coord', 'PAS_h', 'PAS_txrate',  'num_introns',
                            'eta_val', 'insertsize_max', 'insertsize_min', 'read_length',
                            'lambda_init', 'burst_size', 'wiggle_room'] # list of class variables that are simple attributes (not dataframe)
        attr_dict = {attr: getattr(cls, attr) for attr in class_simple_atts}
        for key in attr_dict.keys():
            if isinstance(attr_dict[key], np.int64):
                attr_dict[key] = int(attr_dict[key])  # this is needed because json is quite stupid, no going around it
        json_fn = os.path.join(folder, 'Experiment.json')
        json_file = open(json_fn, 'w')
        json.dump(attr_dict, json_file)
        # second, save the gtf_df to csv
        gtf_fn = os.path.join(folder, 'gtf_df.csv')
        cls.gtf_df.to_csv(gtf_fn, sep='\t', index=False, header =True)
        elongf_fn = os.path.join(folder, 'elongf_df.csv')
        cls.elongf_df.to_csv(elongf_fn, sep='\t', index=False, header=True)
        return

    @classmethod
    def load_class_variables(cls, folder):
        """
        Load the class variables from a file
        :param folder: the folder that the file will be loaded from
        :return: None
        """
        # first, load the simple attributes from json
        json_fn = os.path.join(folder, 'Experiment.json')
        json_file = open(json_fn, 'r')
        attr_dict = json.load(json_file)
        cls.set_read_params(eta_val=attr_dict['eta_val'], insertsize_min=attr_dict['insertsize_min'], insertsize_max=attr_dict['insertsize_max'], read_length=attr_dict['read_length'])
        cls.set_burst_init_params(lambda_init=attr_dict['lambda_init'], burst_size=attr_dict['burst_size'], wiggle_room=attr_dict['wiggle_room'])
        # second, load the gtf_df from csv with other attributes related to the gene structure
        gtf_df = pd.read_csv(os.path.join(folder, 'gtf_df.csv'), sep='\t', header=0)
        cls.set_gtf_df(gtf_df)


    def __init__(self, time_point: int, prev_exp = None, next_exp= None, transcripts=None):
        """
        Initialize the experiment object
        :param time_point: time idx of the expeirment
        :param prev_exp: object of the previous experiment
        :param next_exp: object of the next expeirment
        :param transcripts: list of Transcript objects that are in this experiment
        """
        self.time_point = time_point
        self.prev_exp = prev_exp
        self.next_exp = next_exp
        self.trans_df = pd.DataFrame() # should be created by calling
        self.reads_df = pd.DataFrame()  # should be created by calling the get_reads_df function
        self.junction_df = pd.DataFrame()  # df showing the junction read count in this experiment, should be created by calling the count_junction_reads function
        if transcripts is None: # setting it to NONE by default is needed because of something called "mutable default arguments" in python
            self.transcripts = []
        else:
            self.transcripts = transcripts

    def create_random_transcripts(self, tpm, num_total_transcript_millions):
        """
        Based on the desired gene expression levels, generate new transcripts that can start at any time point during the experiment (random end point).
        This function will add new transcripts to the self.transcripts list (which is supposedly empty in regular cases when this function is called)
        :param tpm: the desired gene expression level at this time point, in TPM
        :param num_total_transcript_millions: number of millions of transcripts that this cell/cells generated through transcription
        :return: generate new transcripts that can end at random points along the gene by the time that we measure this experiment--> self.transcripts are updated
        """
        # the reaason why we need the total number of transcript is because:
        # - TMP_i = (RPK_i / \sum_j-1^{num_gene} RPK_j) * 10^6--> relative abundance of transcription in gene i
        # --> TPM_i/10^6 = relative abundance of transcription in gene i compared to other genes
        # - RPK_i = number of reads mapped to gene i/ length of gene i --> control for hte fact that there are more reads in a longer gene than a shorter gene
        # If we want to simulate transcripts of a gene, given its TPM, we need to know how many transcript in total were generated from all the genes, this is what num_total_transcript_millions is for.
        # N_i = TPM_i/10^6 * num_total_transcript = TPM_i * num_total_transcript_millions--> number of transcripts of gene i
        num_transcripts_to_start = tpm * num_total_transcript_millions
        gene_start = Experiment.gtf_df['start'].iloc[0]  # the start of the gene
        max_transcript_end = np.max(Experiment.gtf_df['end'])  # due to how the function read_gtf_file is written, the last feature in gtf_df is the PAS and RTR, so the end of the last feature is the end of the gene plus the RTR length
        # randomly sample the end sites of the transcripts, these are ABSOLUTE coordinates of the end sites
        endpoints = np.random.randint(gene_start, max_transcript_end, num_transcripts_to_start)
        num_exist_trans = len(self.transcripts)  # usually should be 0
        # print(num_exist_trans)
        for trans_idx, endpoint in enumerate(endpoints):
            self.transcripts.append(Transcript(trans_idx=num_exist_trans+trans_idx, endpoint=endpoint, simulate_cleavage=Experiment.simulate_cleavage)) # new transcript with random end site
        return

    def elongate_existing_transcripts(self, time):
        """
        Elongate the transcripts that are currently in the cell
        :param time: the time period (in minutes) that we want to elongate the transcripts
        :return: a list of elongated transcripts (separate transcript objects from transcript objects in self.transcripts). These transcripts can then be added to a new Experiment object
        """
        elongated_transcripts = []
        for trans_idx, transcript in enumerate(self.transcripts):
            new_stop, new_splice_df, is_cleaved, is_degrade = transcript.elongate(time)
            elongated_transcripts.append(Transcript(trans_idx = trans_idx, endpoint=new_stop, splice_df = new_splice_df, set_cleaved=is_cleaved, set_degrade=is_degrade, simulate_cleavage=Experiment.simulate_cleavage)) # this step aready includes checking transcript maturity.
        return elongated_transcripts

    def find_equilibrium_init(self, max_time: int):
        """
        Beta testing this function for now
        :param max_time: the maximum time that we want to simulate the system
        :return:
        """
        self.init_bursting_transcripts(max_time) # create new transcripts that are generated in bursts, assuming that we let the system rn through its course until it gets to equilibrium
        # for each transcript that were created since the begining of time, we can evaluate whether the transcript has matured, and hence we will consider them as being "degraded".
        # We really only care about nascent mRNA molecules here.
        for trans_idx, transcript in enumerate(self.transcripts):
            transcript.evaluate_transcript_mature()
        return

    def wash_mature_transcripts(self):
        """
        Remove the mature transcripts from the cell
        This function is useful when we simulate transcripts at t=0 with the init_bursting_transcripts function, and then we let the system run through its course until it gets to equilibrium (theoretically, it is when number of newly init transcripts is equal to number of transcripts that become mature). At equilibrium, we only care about nascent transcripts, so we will remove the mature transcripts from the cell
        :return: changes to self.transcripts to include only nascent transcripts
        """
        nascent_transcripts = [transcript for transcript in self.transcripts if (not transcript.is_degrade and not transcript.is_mature)]
        for idx, nascent_trx in enumerate(nascent_transcripts):
            nascent_trx.trans_idx = idx # reset the transcript index
        # clear out the memory of self.transcripts
        del (self.transcripts)
        # set it back to the list with only nascent transcripts
        self.transcripts = nascent_transcripts
        return

    def init_bursting_transcripts(self, time:int):
        """
        Initialize a new transcript that starts during the time period speciifed by time, lambda_init and burst_size
        Here, I am trying to simulate the events of transcription initiation as a bursting model. Assumptions:
        - Initiation of transcription occurs in bursts. When a burst happens, burst_size new mRNA are created.
        - The time between bursts is exponentially distributed with rate lambda_init events per minute.
        :param time: the time period (in minutes) from previous point to current point that we want to simulate new transcripts
        :param lambda_init: the rate of transcription initiation bursts (lambda_init events per minute)
        :param burst_size: the number of transcripts that are generated in each burst
        :param wiggle_room: If we are at time t where burst just happened. Next, we sample the time interval between the last burst (at t) and the next burst (exponential with scale 1/lambda_init), then the trnascripts are created as part of the next burst will be scattered around time [t +- wiggle_room * time_interval]. This is more reasonable because we do not want to have all the transcripts created at the EXACTLY the same time (burst moment).
        :return: Add new transcripts to the existing transcripts list --> modify self.transcripts
        """
        # now we will simulate events of transcription initiation as a Poisson process with rate avg_bursts, which means time between each burst is exponential with rate 1/avg_bursts
        curr_time = 0
        if Experiment.burst_size == 0:  # if burst_size is 0, then no new transcripts are created
            return
        while curr_time < time:
            time_till_next_burst = np.random.exponential(scale=1/Experiment.lambda_init)  # time till next burst
            # create new burst_size transcripts that are actually created during time period [curr_time+ time_till_next_burst +- wiggle_room * time_till_next_burst]
            curr_time += time_till_next_burst # this is the exact time of the burst
            # now we will sample the time that each transcript is created during this burst period's wiggle time
            wiggle_time = np.random.uniform(low=-Experiment.wiggle_room*time_till_next_burst, high=Experiment.wiggle_room*time_till_next_burst, size=Experiment.burst_size)
            wiggle_time = curr_time + wiggle_time # this is the time that each transcript is created
            # now we will calculate the end site of the transcript based on how long it has been since created
            # right now, wiggle time means the time since o that this transcript is created. If we want to ask how long is this transcript when we are at t=time, we will have call calculate_enlongated_endsite on time_since_prev = time-wiggle_time, clipped at 0
            time_since_prev = np.clip(time - wiggle_time, 0, np.inf) # time_since_prev is the time since the transcript is created till time t. The name is not great but it corresponds to variable name in calculate_elognaed_endsite
            vectorized_func = np.vectorize(lambda x: calculate_enlongated_endsite(prev_stop=Experiment.gene_start, elongf_df=Experiment.elongf_df, time_since_prev=x))
            trans_endpoints = vectorized_func(time_since_prev)  # apply function to each element in the array wiggle_time --> endpoint of transcript at the time that we start recording the experiment.
            # now we will create the transcripts
            curr_num_trans = len(self.transcripts)
            for trans_idx, endpoint in enumerate(trans_endpoints):
                self.transcripts.append(Transcript(trans_idx = curr_num_trans+trans_idx, endpoint=endpoint, simulate_cleavage=Experiment.simulate_cleavage)) # new transcript with random end site
        return

    def get_trans_df(self):
        """
        Get the dataframe of transcripts at this time point
        :return: dataframe of transcripts. Required columns: trans_idx, endpoint, is_degrade, time_idx, intron{}_spliced for each intron
        """
        if not self.trans_df.empty: # it means it has been created before, so we do not need to do anything
            return self.trans_df
        df = pd.DataFrame()
        df['trans_idx'] = np.arange(len(self.transcripts))
        df['endpoint'] = [transcript.endpoint for transcript in self.transcripts]
        df['is_degrade'] = [transcript.is_degrade for transcript in self.transcripts]
        df['is_cleaved'] = [transcript.is_cleaved for transcript in self.transcripts]
        df['time_idx'] = self.time_point
        for intron_index in range(Experiment.num_introns):
            df['intron{}_spliced'.format(intron_index)] = list(map(lambda x: x.is_intron_spliced(intron_index), self.transcripts))
        self.trans_df = df
        return df

    def get_reads_df(self):
        """
        Get the dataframe of reads at this time point
        :return: create self.reads_df. required_columns: ['abs_start', 'abs_end', 'subtract_start', 'subtract_end', 'trans_idx']
        """
        if not self.reads_df.empty:  # reads_df is not empty. It has already been created, this function is already called before, so we do not need to do anything
            return
        # the reads_df is empty, so we will create it
        self.reads_df = pd.DataFrame()
        reads_df_list = []
        for trans_idx, transcript in enumerate(self.transcripts):
            transcript.determine_reads()
            # if the reads_df is not empty, add it to the reads_df_list.
            # there are cases where the transcripts are short, and the fragments within the transcripts do not pass the length requirement and so the reads_df is empty
            if not transcript.reads_df.empty:
                reads_df_list.append(transcript.reads_df)
        self.reads_df = pd.concat(reads_df_list, ignore_index=True) # ignore_index is needed to avoid the case where multiple read_df have the same index
        # for numeric columns, we will convert them to integers
        self.reads_df = self.reads_df.apply(helper.convert_to_int_if_numeric)
        return

    ########### below are functions that allow me to save and load data of the experiment object into a file ###########
    def save_experiment(self, folder):
        """
        Save the experiment object into a file
        :param folder: the folder that the file will be saved to. It should be <class_folder>/Exp_<time_point> where class_folder is the folder that contains the Experiment.json file (see function save_class_variables)
        :return: None
        """
        helper.make_dir(folder)
        # because the only simple attributes of the experiment object are time_point we will NOT save them to json
        # instead, we will assume that the folder of the experiment is such that
        # |__ class_folder
        # |   |__ Experiment.json
        # |   |__ gtf_df.csv
        # |__ Exp_<time_point>
        # |   |__ transcripts.csv.gz
        # |   |__ reads.csv.gz
        trans_fn = os.path.join(folder, 'transcripts.csv.gz')
        reads_fn = os.path.join(folder, 'reads.csv.gz')
        trans_df = self.get_trans_df()
        trans_df.to_csv(trans_fn, sep = '\t', header=True, index=False, compression='gzip')
        self.get_reads_df()
        self.reads_df.to_csv(reads_fn, sep = '\t', header=True, index=False, compression='gzip')
        return

    def load_experiment(self, folder, create_trans=False, time_point=None):
        """
        Load the experiment object from a file
        :param folder: the folder that the file will be loaded from
        :param create_trans: if True, then create the transcripts list from the transcripts.csv.gz file. If not (default), dont --> save time and space
        :param time_point: the time point of this experiment that we are loading
        :return: None
        """
        if time_point is None:
            time_point_from_path = os.path.basename(os.path.normpath(folder)).split('_')[1] # get the time point from the folder name
            self.time_point = int(time_point_from_path) # set the time point of this experiment
        else:
            self.time_point = time_point
        trans_fn = os.path.join(folder, 'transcripts.csv.gz')
        reads_fn = os.path.join(folder, 'reads.csv.gz')
        self.trans_df = pd.read_csv(trans_fn, header=0, index_col=None, sep='\t')
        if create_trans:   # note I highly recommend that users load_experiment only when they only need to see trans_df and reads_df (create_trans=False), and not other transcript manipulation because this function was not throughly tested to fit with other functions such as elongate_existing_transcripts, etc.
            self.transcripts = []
            for row_idx, row in self.trans_df.iterrows():
                self.transcripts.append(Transcript(trans_idx = row_idx, endpoint=row['endpoint'], set_cleaved=row['is_cleaved'], set_degrade=row['is_degrade']))
        self.reads_df = pd.read_csv(reads_fn, header=0, index_col=None, sep='\t')
        return

    ########### Below are functions that are used to analyze the behaviors of the simulation (so that we can plot out distribution of transcripts/reads) ###########
    def tag_reads_by_timepoint(self, endpoint_df):
        """
        Given the reads that we obtain in this experiment, and a df showing the endpoint of transcript at different points in time, we would like to tag each read with the time point that it is generated/elongated (have the tags of the timepoints based on SLAM-seq experiment)
        :param endpoint_df: df with rows: transcripts, columns: time_idx, values: endpoint of the transcript at each time point
        Important assumption: endpoint_df's columns are 0, ... num_timepoints-1, and these columns correspond to experiments done in chronoligcal time order
        :return: add a column to self.reads_df called tagged_time, which is the time_idx that the read will be assigned to if they are sequenced ana mapped as part of the SLAM-seq pipeline
        """
        # for each read, determine the time point that the transcript was created/elongated (if the reads overlap with a region that include the end point of transcripts at two different time points, then assign the read to the later time point)
        # this step involves determining: For each read, find the FIRST time point t such that the transcript's endpoint at t is >= the read's abs_end.
        # Checks that all corner cases are covered:
        # - If the reads is a EI/II junction: transcript unspliced at intron I in the last time point.
        # - If the read is a EE junction: transcript spliced at intron I in the last time point, and even if part of the reads are tagged to time point t and part of it is tagged to time point t+j, the read is still tagged to time point t+j, even if the transcript is spliced I at a later time point that t+j
        # - If the read overlaps an exon, then it is obvious that it is assigned to the correct time point given our algorithm
        # - If the transcript is elongated at t past PAS, and is then cleaved at t+j, then the read will be tagged as t (not t+j)
        self.get_reads_df() # if the reads_df is empty, then we will create it
        num_timepoints = len(endpoint_df.columns)
        self.reads_df = self.reads_df.merge(endpoint_df, left_on = 'trans_idx', right_index=True, how='left')   # for each read, add columns showing the endpoint of the corresponding transcript at each time point
        # for each read (row), there are columns: 0, ... num_timepoints-1,
        # find the index of the FIRST timepoint that has endpoint >= read's abs_end
        self.reads_df['tagged_time'] = self.reads_df.apply(lambda row: get_reads_tagged_time(row, num_timepoints), axis=1)  # if the read is sequenced and mapped, then the tagged_time is the time_idx of the experiment, otherwise it is nan
        self.reads_df.drop(np.arange(num_timepoints), axis=1, inplace=True)  # drop the columns 0, ... num_timepoints-1 because we do not need them anymore
        return

    def count_junction_reads(self, with_tagged_time=False, endpoint_df=None):
        """
        for each intron, calcualte the number of junction reads in this experiment
        :param with_tagged_time: whether or not we should count junction reads stratified by with the time point that the read is generated/elongated.
        :param endpoint_df: df with rows: transcripts, columns: time_idx, values: endpoint of the transcript at each time point. This should be provided when with_tagged_time is True
        :return: a df showing the number of junction reads. rows: feature (intron_1, intron_2), columns: EE_reads, ie_reads (here, EI and IE are grouped to the same category)
        """
        # the junction_df is empty, or that the current format of the junction_df is not what you would like it to be (based on with_tagged_time flag), so we will create (or recreate) it
        self.get_reads_df()
        ee_reads_columns_to_keep = ['feature', 'ee_reads']
        reads_df = self.reads_df # if with_tagged_time is False, then we will use the reads_df as is
        merge_columns = ['feature'] # if with_tagged_time is False, then we will merge columns of ee_reads and ie_reads on only these columns later
        if with_tagged_time:
            assert endpoint_df is not None, 'endpoint_df is not provided to function count_junction_reads when param with_tagged_time is True'
            self.tag_reads_by_timepoint(endpoint_df)
            reads_df = self.reads_df.groupby('tagged_time')
            ee_reads_columns_to_keep = ['feature', 'ee_reads', 'tagged_time']
            merge_columns = ['feature', 'tagged_time']  # if with_tagged_time is True, then we will merge columns of ee_reads and ie_reads on these columns later
        # exon-exon junction reads are reads that have subtract_start and subtract_end correpsonding to the start and end of the intron
        ee_reads = reads_df[['subtract_start', 'subtract_end']].value_counts().to_frame().reset_index()
        ee_reads = ee_reads.merge(Experiment.gtf_df, left_on=['subtract_start', 'subtract_end'],
                                            right_on=['start', 'end'], how='left').rename(columns={'count': 'ee_reads'})  # will only merge on rows that are intron in gtf_df
        ee_reads = (ee_reads[ee_reads_columns_to_keep]).dropna(axis=0)  # drop the rows that are not junction reads (not in the gtf)
        # now, ee_reads has columns: feature, ee_reads if with_tagged_time is False. If with_tagged_time is True, then ee_reads has columns: tagged_time, feature, ee_reads
        # exon-intron junction reads are reads that have an exon-intron (or intron-exon) boundary within the length of the read
        # first find all the IE or EI boundary, ordered in ascending order along the length of the gene
        intron_df = Experiment.gtf_df[Experiment.gtf_df['is_intron']][['feature', 'start', 'end']]  # already sorted in ascending order
        ie_reads = self.reads_df.apply(lambda row: determinte_IE_junction(row, intron_df), axis=1).to_frame()  # df with one column: 0 --> None if not a IE junction, otherwise the feature that this read is a junction for (intron_1, intron_2 etc.)
        if with_tagged_time:
            ie_reads['tagged_time'] = self.reads_df['tagged_time']  # add another column showing the time that each read is tagged to
        ie_reads = ie_reads.value_counts().reset_index(drop=False).rename(columns={0: 'feature', 'count': 'ie_reads'})  # if with_tagged_time is True, then ie_reads has columns: tagged_time, feature, ie_reads. Otherwise, ie_reads has columns: feature, ie_reads
        # merge ee_reads and ie_reads
        junction_df = ee_reads.merge(ie_reads, on=merge_columns, how='outer').fillna(0)
        junction_df['time_idx'] = self.time_point  # the time point of this experiment
        return junction_df

    def get_ee_junction_df(self, with_tagged_time=False, endpoint_df=None):
        """
        Get the exon-exon junction reads in this experiment
        OUtput df will only contains rows corresponding to different reads
        We do not just want the count here, we want the whole reads
        :param with_tagged_time: whether or not we should count junction reads stratified by with the time point that the read is generated/elongated.
        :param endpoint_df: df with rows: transcripts, columns: time_idx, values: endpoint of the transcript at each time point. This should be provided when with_tagged_time is True
        :return: a df showing the number of exon-exon junction reads. rows: feature (intron_1, intron_2), columns: EE_reads
        """
        from .junction_reads import add_time_brakpoint_to_EE_junction_reads
        if with_tagged_time:
            assert endpoint_df is not None, 'endpoint_df is not provided to function get_ee_junction_df when param with_tagged_time is True'
            if 'tagged_time' not in self.reads_df.columns:
                self.tag_reads_by_timepoint(endpoint_df)
        juncReads_df = (self.reads_df[(self.reads_df['subtract_start'] != self.reads_df['subtract_end'])]).copy().reset_index().rename(columns={'index': 'read_index'})  # exon-exon junction reads
        juncReads_df['intron'] = juncReads_df.merge(Experiment.gtf_df, left_on=['subtract_start', 'subtract_end'], right_on=['start', 'end'], how='left')['feature']
        juncReads_df = add_time_brakpoint_to_EE_junction_reads(juncReads_df, num_timepoints=self.time_point+1)
        self.juncReads_df = juncReads_df
        return juncReads_df

