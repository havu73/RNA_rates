import os
import pandas as pd
import numpy as np
from .experiment import Experiment
from .transcripts import Transcript
import matplotlib.pyplot as plt
plt.style.use('default')
plt.rcParams['figure.facecolor'] = 'white'
from . import helper
ONE_KB=1000
DEFAULT_SIM_FEAT_LEN = 5000 # length of one feature in the simulation
SEED = 9999
np.random.seed(SEED)
print ('importing simulate_multiple_experiments')


def get_feature_names_from_nExons(nExons):
    """
    Given the number of exons, return the feature names
    :param nExons: number of exons
    :return: a list of feature names, it should correspond to a gene with the structure: exon_1, intron_1, exon_2, intron_2, ..., exon_nExons.
    """
    feature_names = []
    for i in range(nExons):
        feature_names.append('exon_{}'.format(i+1))
        if i < nExons - 1:
            feature_names.append('intron_{}'.format(i+1))
    return feature_names

def create_control_gtf_df(nExons:int=3, intron_h:float=1., PAS_h:float=1., SIM_FEAT_LEN=DEFAULT_SIM_FEAT_LEN, RTR=DEFAULT_SIM_FEAT_LEN):
    """
    Create a gtf_df for a control experiment, with constant features
    :param nExons: number of exons in the gene, the gene is assumed to have the structure exon_1, intron_1, exon_2, intron_2, ..., exon_nExons, PAS, RTR
    :param intron_h: fixed at 1 so that users can decide the fold that they want to change the intron_h
    :param PAS_h: fixed at 1 so that users can decide the fold that they want to change the PAS_h
    :param SIM_FEAT_LEN: length of each feature in the simulation
    :param RTR: length of the readthrough region
    :return:
    """
    # first create a gtf of your choice
    gtf_df = pd.DataFrame()
    gtf_df['feature'] = get_feature_names_from_nExons(nExons=nExons)
    gtf_df['start'] = SIM_FEAT_LEN * np.arange(0,len(gtf_df.feature))
    gtf_df['end'] = gtf_df['start'] + SIM_FEAT_LEN
    gtf_df.loc[gtf_df.shape[0]] = ['PAS', gtf_df.end.iloc[-1], gtf_df.end.iloc[-1]+1]
    gtf_df.loc[gtf_df.shape[0]] = ['RTR', gtf_df.end.iloc[-1], gtf_df.end.iloc[-1] + RTR]
    gtf_df['length'] = gtf_df['end'] - gtf_df['start']
    gtf_df['txrate'] = 1.0 # 1KB per minute
    gtf_df['is_intron'] = gtf_df['feature'].apply(lambda x: x.split('_')[0]=='intron')  # exon_1, intron_1 etc. --> True or False whether this is an intron or not
    if 'intron_h' not in gtf_df.columns:
        gtf_df['intron_h'] = gtf_df['is_intron'].apply(lambda x: intron_h if x else 0.)  # if this is an intron, then assign the intron_h, otherwise assign 0 (exon or irrelebant feature to splicing)
    gtf_df['is_PAS'] = gtf_df['feature'].apply(lambda x: x=='PAS')
    if 'PAS_h' not in gtf_df.columns:
        gtf_df['PAS_h'] = gtf_df['is_PAS'].apply(lambda x: PAS_h if x else 0.)  # if this is a PAS, then assign the PAS_h, otherwise assign 0 (irrelebant feature to cleavage)
    return gtf_df

def create_variable_gtf_df(nExons:int=3, elong_fold_list:list=None, length_fold_list=[1.]*5, intronH_fold_list:list=[1.]*5, PAS_h:int=1, SIM_FEAT_LEN:int = DEFAULT_SIM_FEAT_LEN, RTR:int = DEFAULT_SIM_FEAT_LEN):
    """
    create a gtf_df such that each features can have variable elongation rate and length
    :param nExons: # of exons in the gene. Gene structure: exon_1, intron_1, exon_2, intron_2, ..., exon_nExons, PAS, RTR
    :param elong_fold_list: If None, it means the users may choose not to specify elongation rates through gtf_df (they have elongf_df separately). showing the fold to change elongation rate of each exon and intron in the gene. Here, fold change compared to the control elongation rate.
    :param length_fold_list: a list of length nExons*2-1, showing the fold to change the length of each exon and intron in the gene. Here, fold change compared to the control length.
    :param intron_h: control intron_h (splicing half-life)
    :param PAS_h: control PAS_h (cleavage half-life)
    :param SIM_FEAT_LEN: length of each feature (exon/intron) in the simulation
    :param RTR: length of the readthrough region
    :return:
    """
    # assert that the length of elong_fold_list and length_fold_list is correct
    assert len(length_fold_list) == nExons*2-1, 'length of length_fold_list is not correct. Should be nExons*2-1'
    assert len(intronH_fold_list) == nExons*2-1, 'length of intron_h is not correct. Should be nExons*2-1'
    # create a control gtf_df
    gtf_df = create_control_gtf_df(nExons=nExons, SIM_FEAT_LEN=SIM_FEAT_LEN, RTR=RTR)
    # modify the gtf_df: change each feature's length
    gtf_df.loc[:(len(length_fold_list)-1),'length'] = gtf_df['length'] * pd.Series(length_fold_list)
    gtf_df.loc[:(len(intronH_fold_list)-1),'intron_h'] = gtf_df['intron_h'] * pd.Series(intronH_fold_list)
    if elong_fold_list is not None:
        assert len(elong_fold_list) == nExons*2-1, 'length of elong_fold_list is not correct. should be nExons*2-1'
        gtf_df.loc[:(len(elong_fold_list)-1),'txrate'] = gtf_df['txrate'] * pd.Series(elong_fold_list)
    # modify the end point of each feature so that they align with the lengths
    gtf_df['start'] = gtf_df.loc[0, 'start'] + np.cumsum(gtf_df['length']) - gtf_df['length']
    gtf_df['end'] = gtf_df['start'] + gtf_df['length']
    return gtf_df


def create_elongf_df(elong_fold_list:list=[1.]*5, length_fold_list=[1.]*5, SIM_FEAT_LEN:int=DEFAULT_SIM_FEAT_LEN):
    """
    Create the elongf_df from the gtf_df. The elongf_df is a dataframe that shows the elongation rates of each feature in the gene. It has the columns: start, end, txrate, feature
    :param elong_fold_list: a list showing the folds of elongation rates for each feature
    :param length_fold_list: a list showing the folds of length for each bin (each of which can have different elongation rates)
    :return: elongf_df
    """
    elongf_df = pd.DataFrame()
    elongf_df['length'] = SIM_FEAT_LEN * pd.Series(length_fold_list)
    elongf_df['txrate'] = pd.Series(elong_fold_list)  # 1KB per minute
    elongf_df['start'] = np.cumsum(elongf_df['length']) - elongf_df['length']
    elongf_df['end'] = elongf_df['start'] + elongf_df['length']
    # add one more row corresponding to the region after the last feature
    elongf_df.loc[len(elongf_df)] = {'txrate': elong_fold_list[-1], 'start': elongf_df['end'].iloc[-1], 'end': np.inf, 'length': np.inf}
    return elongf_df

def align_gtf_df_with_elongf_df(gtf_df, elongf_df):
    '''
    This function should try to do some quality control for gtf_df and elongf_df, and align the two dataframes together so that they jointly specify the required features of the gene
    :param gtf_df:
    :param elongf_df:
    :return:
    '''
    assert elongf_df is not None, 'align_gtf_df_with_elongf_df: elongf_df should not be None'
    # 2. gtf_df should end with PAS and RTR
    assert gtf_df['feature'].iloc[-2] == 'PAS' and gtf_df['feature'].iloc[-1] == 'RTR', 'gtf_df should end with PAS and RTR'
    # 3. the last feature of gtf_df (RTR) should have length exactly equal to the length of 5 minutes of elongation rate. This is just a practical decision in the simulation
    last_txrate = elongf_df['txrate'].iloc[-1]
    print('last_txrate:', last_txrate)
    # last_length = int(5 * last_txrate * ONE_KB)  # in gtf_df, length is in bp and txrate is in 1KB per minute
    # gtf_df.loc[gtf_df.index[-1], 'length'] = last_length
    # # modify the end point of each feature so that they align with the lengths
    # gtf_df['start'] = gtf_df.loc[0, 'start'] + np.cumsum(gtf_df['length']) - gtf_df['length']
    # gtf_df['end'] = gtf_df['start'] + gtf_df['length']
    # 1. elongf_df should end with np.inf if it exists
    assert elongf_df['end'].iloc[-1] == np.inf, 'elongf_df should end with np.inf'
    # 4. Get rid of txrate column in gtf_df, to avoid confusion since we already have elongf_df
    try:
        gtf_df.drop('txrate', axis=1, inplace=True)
    except:
        pass
    return gtf_df, elongf_df



def generate_exp_given_one_gtf(gtf_df, elongf_df = None, save_folder= None, label_time = [0,5,10,15], target_exp=5, num_total_transcript_millions = 100, lambda_init = 0.5 , burst_size = 5, wiggle_room = 0.3, frag_func:str = 'weibull', eta_val=helper.DFT_ETA_VALUE, unif_avg_frag_len= helper.DFT_UNIF_FRAG_LEN, insertsize_min=helper.DFT_INSERTSIZE_MIN,insertsize_max=helper.DFT_INSERTSIZE_MAX, read_length=helper.DFT_READ_LENGTH, pair_end= False, simulate_cleavage=True, PDB=False, max_time_for_equilibrium=1000):
    """
    This function generates a list of experiments corresponding to multiple timepoints, given a gtf_df
    :param gtf_df: gtf_df showing the features of the gene
    :param elongf_df: elongf_df showing the elongation rates of the gene. If None, the functions will then implement elongf_df to be exactly similart to the gtf_df. Columns that will be used: start, end, txrate, feature?
    :param save_folder: folder to save the experiments . The file structure will be as follows:
    |__ save_folder
    |   |__ Experiment.json
    |   |__ gtf_df.csv
    |__ Exp_<time_point>
    |   |__ transcripts.csv.gz
    |   |__ reads.csv.gz
    :param label_time: a list showing the label time points that we will simulate the experiments at
    :param target_exp: the target expression level (in TPM) that we want to simulate
    :param num_total_transcript_millions: the total number of transcripts (in millions) that we want to simulate
    :param lambda_init: The lambda param for exp(1/lambda) distribution for the time between burst events of transcription initiation
    :param burst_size: The number of transcripts that are created in a burst event
    :param wiggle_room: The wiggle room for the burst events (please see the documentation of Experiment.init_bursting_transcripts() for more details). If not sure, leave it as default
    :param simulate_cleavage: whether to simulate cleavage or not in this experiment
    :return: a list of experiments
    """
    print('inside generate_exp_given_one_gtf')
    assert frag_func in helper.FRAGMENT_DIST_LIST, 'frag_func should be either weibull or uniform'
    Transcript.set_gtf_df(gtf_df)
    Transcript.set_elongf_df(elongf_df)
    Transcript.set_read_params(pair_end=pair_end, eta_val=eta_val, insertsize_min=insertsize_min,
                               insertsize_max=insertsize_max, read_length=read_length, frag_func=frag_func, unif_avg_frag_len=unif_avg_frag_len)
    Experiment.set_gtf_df(gtf_df)
    Experiment.set_elongf_df(elongf_df)
    Experiment.set_simulate_cleavage(simulate_cleavage)
    Experiment.set_read_params(pair_end=pair_end, eta_val=eta_val, insertsize_min=insertsize_min,
                               insertsize_max=insertsize_max, read_length=read_length)
    Experiment.set_burst_init_params(lambda_init=lambda_init, burst_size=burst_size, wiggle_room=wiggle_room)
    exp0 = Experiment(time_point=0, prev_exp=None, next_exp=None)  # create the first experiment
    # create randomly-ended transcripts in this experiment (this represents the state of the system at t=0 with transcripts that we generated at various time points in the past)
    if not PDB:  # if we do not do PBD, the first time step is created based on letting the system reach equilibrium first
        exp0.find_equilibrium_init(max_time= max_time_for_equilibrium) # run the bursting init moel for a long time to produce new transcripts continuously. Many of the transcripts have become mature and will be irrelevant to our analyses
        exp0.wash_mature_transcripts() # wash the mature transcripts away, we dont care for those. this step will reset the transcript indices from 0 to # nascent transcripts
    else: # if we DO PDB experiment --> the first tiem step is simply 0, and we will not create any new transcripts
        exp0.create_random_transcripts(tpm=0, num_total_transcript_millions=num_total_transcript_millions)
    print('before adding new experiments')
    print(len(exp0.transcripts))
    exp_list = [exp0]
    for time_point in range(1, len(label_time)):
        prev_exp = exp_list[-1]  # transcript from previous time point
        elongated_transcripts = prev_exp.elongate_existing_transcripts(time=label_time[time_point] - label_time[time_point - 1])  # elongate the transcripts from the previous time point
        new_exp = Experiment(time_point=time_point, prev_exp=prev_exp, next_exp=None,
                             transcripts=elongated_transcripts)  # put those elongated transcripts into the new experiment
        new_exp.init_bursting_transcripts(time=label_time[time_point] - label_time[time_point - 1])  # create new transcripts between previous and current time point
        print(len(new_exp.transcripts))
        prev_exp.next_exp = new_exp  # link the two experiments together
        # new_exp.get_reads_df()  # get the reads_df for the new experiment
        exp_list.append(new_exp)
    # Now write the experiment into a folder and return the experiment list
    if save_folder != None:
        Experiment.save_class_variables(folder=save_folder)
        # exp_folder_list = list(map(lambda x: os.path.join(save_folder, 'Exp_{}'.format(x)), range(len(exp_list))))
        # for exp_indx , exp_folder in enumerate(exp_folder_list):
        #     exp_list[exp_indx].save_experiment(exp_folder)
    return exp_list


def read_one_exp_sequence(save_folder, num_exp=3):
    """
    Given that the function generate_exp_given_one_gtf() has been run with a save_folder, this function reads the experiments into list of experiments objects, which can be helpful in analyzing the behavior of the system
    :param save_folder: where the experiments have been saved when they were created (important data should be in transcripts.csv.gz and reads.csv.gz)
    |__ save_folder
    |   |__ Experiment.json
    |   |__ gtf_df.csv
    |__ Exp_<time_point>
    |   |__ transcripts.csv.gz
    |   |__ reads.csv.gz
    :return:
    """
    Experiment.load_class_variables(folder=save_folder)
    exp_list = []
    for exp_idx in range(num_exp):
        exp = Experiment(time_point=exp_idx, prev_exp=None, next_exp=None)
        exp.load_experiment(folder=os.path.join(save_folder, 'Exp_{}'.format(exp_idx)))
        exp_list.append(exp)
    return exp_list


def read_multiple_exp_sequences(save_folder):
    for i in range(1, 4):
        print('Generating plots and results for {}'.format(i))
        exp_folder = os.path.join(save_folder, 'exon_r_fold_{}'.format(i))
        read_one_exp_sequence(save_folder=exp_folder)
    return

