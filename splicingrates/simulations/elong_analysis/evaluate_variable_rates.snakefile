import numpy as np
import pandas as pd
from itertools import product

big_folder = '/gladstone/engelhardt/lab/hvu/RNA_rates/elong_rates/pair_end_reads/'
subfolder_list = ['constant', 'variable/bin0.2_fold2',  'variable/bin1.0_fold2', 'variable/bin0.2_fold4' ,'variable/bin1.0_fold4', 'variable/bin0.5_fold2', 'variable/bin0.5_fold4']
design_fn = '/gladstone/engelhardt/lab/hvu/RNA_rates/elong_rates/pair_end_reads/constant/design_matrix.csv'
def num_lines_in_file(fn):
    df = pd.read_csv(fn, header =0, index_col=None, sep = '\t')
    return df.shape[0]

num_jobs = num_lines_in_file(design_fn)

jobs_to_care = range(num_jobs)
rule all:
    input:
        expand(os.path.join(big_folder,  '{subfolder}', 'result_{job_idx}', 'pred_h.csv.gz'), job_idx = jobs_to_care, subfolder = subfolder_list)



def obtain_params_from_design_matrix(wildcards):
    '''
    This function widll parse a row from the design matrix into a string that can be used as command arguments into the python script
    Input looks like:
    lambda_init  burst_size  insertsize_min  insertsize_max  read_length  seed elong_fold_list length_fold_list                           output_fn
0            2          10              -1              -1           -1     9  (0.06, 0.4, 0.06)    (0.3, 2.0, 0.3)  splice_est/spliceH_est/noReads/result_0.csv
    Output looks like:
    '--lambda_init 2 --burst_size 10 --insertsize_min -1 --insertsize_max -1 --read_length -1 --seed 9 --elong_fold_list 0.06 0.4 0.06 --length_fold_list 0.3 2.0 0.3 --output_fn splice_est/spliceH_est/noReads/result_0.csv'
    Function written by ChatGPT and tested by Ha Vu
    '''
    design_fn = f'{wildcards.folder}/design_matrix.csv'
    design_df = pd.read_csv(design_fn, header =0, index_col=None, sep = '\t')
    row = design_df.iloc[int(wildcards.job_idx)]
    formatted_string = ""
    for colname, value in row.items():
        if isinstance(value, str) and value.startswith('(') and value.endswith(')'):
            # Remove parentheses and split the values by comma
            value = value.strip('()').replace(',', '')
        if not isinstance(value, (bool, np.bool_)):
            formatted_string += f"--{colname} {value} "
        elif isinstance(value, (bool, np.bool_)) and value:
            formatted_string += f"--{colname} "
        else: # if value is False, we don't include it in the command
            continue
    return formatted_string.strip()

rule run_eval_elong_by_job:
    input:
    output:
        os.path.join('{folder}', 'result_{job_idx}', 'pred_h.csv.gz'),
    params:
        other_params = obtain_params_from_design_matrix,
    shell:
        """
        mkdir -p {wildcards.folder}/result_{wildcards.job_idx}
        command="python evaluate_elongation_est_from_desing.py {params.other_params} "
        echo $command
        eval $command
        """

# example command: python evaluate_elongation_est_from_desing.py --lambda_init 2 --burst_size 10 --insertsize_min -1 --insertsize_max -1 --read_length -1 --seed 9999 --constant_elong_rate 0.8375 --length_fold_list 3.35 3.35 3.35 --time_traverse_gene 12.0 --output_folder /gladstone/engelhardt/lab/hvu/RNA_rates/elong_rates/constant/result_7