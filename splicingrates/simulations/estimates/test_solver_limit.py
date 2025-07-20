import os
import numpy as np
import scipy
import helper
import estimate_elongation as estElong
import pandas as pd
from transcription import simulate_multiple_experiments as sim
from transcription import elongation_calculation as elong
ONE_KB=1000
SIM_FEAT_LEN = 5000 # length of one feature in the simulation
SEED = 9999
np.random.seed(SEED)


'''
This script is created to answer the question: if I can perfectly calculate the distance travelled of a trnascript in m minutes, can I calculate the elongation rates at 200-bp resolution? 
In other word, the question is: Is my solver completely correct if my alogirhtm to calculate the distance travelled is correct?
The asnwer is yes!
'''

def calculate_x0_x1(gtf_df, m =5, convert_to_kb=True):
    '''
    Given the parameters of the gtf_df, we will calculate the exact values of x0 and x1 based on the distance travelled in m minutes
    The function will return two numpy arrays of the same size x0 and x1, which shows the exact start and endpoints of a transcript at the beginning and end of the m-minute interval
    This output will be used to solve the elongation rate problem. This will help us understand the limits of the elongation rate solver.
    :param gtf_df:
    :param m: time interval between two labeling milestones
    :return:
    '''
    gene_start = gtf_df.loc[0]['start']
    gene_length = gtf_df[gtf_df['feature'] == 'PAS']['end'].values[0]
    x0 = np.arange(gene_start, gene_length, 1)
    vectorized_function = np.vectorize(lambda x: elong.calculate_enlongated_endsite(x, gtf_df=gtf_df, time_since_prev=m))
    x1 = vectorized_function(x0)
    if convert_to_kb:
        x0 = x0 / ONE_KB
        x1 = x1 / ONE_KB
    return x0, x1

def get_endpoints_for_h_est(gtf_df, binsize=200, covert_to_KB=True):
    '''
    Given the gtf_df, we will calculate the endpoints of the gene in the simulation
    :param gtf_df:
    :param binsize: in bp
    :param covert_to_KB:
    :return:
    '''
    gene_start = gtf_df.loc[0]['start']
    gene_end = gtf_df[gtf_df['feature'] == 'PAS']['end'].values[0]
    endpoints = np.arange(gene_start, gene_end+1, binsize)
    if covert_to_KB:
        endpoints = endpoints / ONE_KB
    endpoints = np.append(endpoints, np.inf)
    return endpoints

def run_one_experiment(num_exons=10, binsize_kb=0.2):
    # first, randomly create the lengths and elongation rate of the features
    num_features = 2*num_exons - 1
    lengths = np.ones(num_features)*binsize_kb
    elong_rates = np.random.uniform(0.1, 10, num_features)
    gtf_df = sim.create_variable_gtf_df(nExons=num_exons, length_fold_list=lengths, elong_fold_list=elong_rates,
                                        intronH_fold_list=[np.inf] * num_features, SIM_FEAT_LEN=ONE_KB)
    time = 5
    convert_to_kb = True
    x0, x1 = calculate_x0_x1(gtf_df, m=time, convert_to_kb=convert_to_kb)
    x0 = x0[::10]
    x1 = x1[::10]
    endpoints = get_endpoints_for_h_est(gtf_df, binsize=200, covert_to_KB=convert_to_kb)
    h = estElong.calculate_h_one_round(x0, x1, endpoints, time)
    print(gtf_df)
    print(h)
    # duplicate the last value of h to match the length of the gtf_df
    h = np.append(h, h[-1])
    gtf_df['pred_txrate'] = h[:gtf_df.shape[0]]
    return gtf_df

if __name__ == '__main__':
    num_exps = 10
    result_df_list = []
    for i in range(num_exps):
        result_df_list.append(run_one_experiment())
    result_df = pd.concat(result_df_list)
    result_df.to_csv('test_solver_limit.csv', sep='\t', index=False)