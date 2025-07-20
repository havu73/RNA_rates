import numpy as np
import pandas as pd
from itertools import product

plotH_folder = 'splice_est/plotH'
est_spliceH_folder = 'splice_est/spliceH_est/noReads/'
design_fn = 'splice_est/spliceH_est/noReads/design_matrix.csv' # python  create_design_parameters.py --exon_length 0.09 0.14 0.25  --intron_length 3.0 7.0 15.0 --exon_elong  0.02 0.03 0.05 --intron_elong 0.6 1.2 3 --output_folder splice_est/spliceH_est/noReads/ --save_fn splice_est/spliceH_est/noReads/design_matrix.csv  --seed 9 999

def num_lines_in_file(fn):
    df = pd.read_csv(fn, header =0, index_col=None, sep = '\t')
    return df.shape[0]

num_jobs = num_lines_in_file(design_fn)
PBD = False
time=5
num_exons = 2
num_features = num_exons * 2 - 1

#vary_spliceH = [list(np.arange(3, 5, 0.5)),
#                list(np.arange(5, 7, 0.5)),
#                list(np.arange(7, 10, 0.5))]
vary_spliceH = list(np.arange(3, 12, 1))+list(np.arange(13, 21, 2))

length_tiers = [[0.5, 1, 1.5],
                [2, 3.5 ,5],
                [6,9,12],
                [15,22.5, 30]] # in kb
num_length_tiers = len(length_tiers)
vary_elongH = [0.6,0.8,1, 1.2, 2.4, 3.6] # kb/minute

seed_list = [668, 704,  64, 444, 513, 859, 723, 927, 381, 194, 779, 788, 163]

burst_size_list = [10,20,30]
lambda_init_list = [0.5, 1, 2, 3, 4]

rule all:
    input:
        # expand(os.path.join(plotH_folder, 'varyIntron_F{intronIdx}_tier{hTier_idx}_S{seed}.csv'), intronIdx= [1,3], hTier_idx=range(num_spliceH_tiers), seed=seed_list),
        # expand(os.path.join(est_spliceH_folder, 'varyIntron_F1_S{seed}_B{burst_size}_lamb{lambda_init}.csv'), seed=seed_list, burst_size = burst_size_list, lambda_init = lambda_init_list),
        expand(os.path.join(est_spliceH_folder, 'result_{job_idx}.csv'), job_idx = range(num_jobs)),



def find_correct_tier(wildcards):
    return ' '.join(f'{f:.1f}' for f in vary_spliceH[int(wildcards.hTier_idx)])

rule run_calcualte_ratios_junctionReads:
    input:
    output:
        os.path.join(plotH_folder, 'varyIntron_F{intronIdx}_tier{hTier_idx}_S{seed}.csv'),
    params:
        with_PBD = "--PBD" if PBD else "",
        vary_intronH_list = find_correct_tier,
        length_fold_list = ' '.join(['5' for i in range(num_features)]),
        elong_fold_list = ' '.join(['1' for i in range(num_features)]),
    shell:
        """
        command="python  evaluate_splicing.py --vary_intronH_list {params.vary_intronH_list} --length_fold_list {params.length_fold_list} --elong_fold_list {params.elong_fold_list} --feat_idx_to_vary {wildcards.intronIdx} --output_fn {output} {params.with_PBD} --time {time} --seed {wildcards.seed}"
        eval $command
        """

rule run_find_spliceH:
    input:
    output:
        os.path.join(est_spliceH_folder, 'varyIntron_F1_S{seed}_B{burst_size}_lamb{lambda_init}.csv'),
    params:
        with_PBD = "--PBD" if PBD else "",
        length_fold_list = ' '.join(['5' for i in range(num_features)]),
        elong_fold_list = ' '.join(['1' for i in range(num_features)]),
        vary_intronH_list = ' '.join([str(h) for h in vary_spliceH]),
    shell:
        """
        command="python evaluate_splicing.py --vary_intronH_list {params.vary_intronH_list} --length_fold_list {params.length_fold_list} --elong_fold_list {params.elong_fold_list} --feat_idx_to_vary 1 --output_fn {output} {params.with_PBD} --time {time} --seed {wildcards.seed}  --burst_size {wildcards.burst_size} --lambda_init {wildcards.lambda_init}"
        eval $command
        """


def obtain_params_from_design_matrix(wildcards):
    '''
    This function will parse a row from the design matrix into a string that can be used as command arguments into the python script
    Input looks like:
    lambda_init  burst_size  insertsize_min  insertsize_max  read_length  seed elong_fold_list length_fold_list                           output_fn
0            2          10              -1              -1           -1     9  (0.06, 0.4, 0.06)    (0.3, 2.0, 0.3)  splice_est/spliceH_est/noReads/result_0.csv
    Output looks like:
    '--lambda_init 2 --burst_size 10 --insertsize_min -1 --insertsize_max -1 --read_length -1 --seed 9 --elong_fold_list 0.06 0.4 0.06 --length_fold_list 0.3 2.0 0.3 --output_fn splice_est/spliceH_est/noReads/result_0.csv'
    Function written by ChatGPT and tested by Ha Vu
    '''
    design_df = pd.read_csv(design_fn, header =0, index_col=None, sep = '\t')
    row = design_df.iloc[int(wildcards.job_idx)]
    formatted_string = ""
    for colname, value in row.items():
        if isinstance(value, str) and value.startswith('(') and value.endswith(')'):
            # Remove parentheses and split the values by comma
            value = value.strip('()').replace(',', '')
        formatted_string += f"--{colname} {value} "
    return formatted_string.strip()

rule run_spliceH_by_job:
    input:
    output:
        os.path.join(est_spliceH_folder, 'result_{job_idx}.csv'),
    params:
        with_PBD = "--PBD" if PBD else "",
        other_params = obtain_params_from_design_matrix,
        vary_intronH_list = ' '.join([str(h) for h in vary_spliceH]),
    shell:
        """
        command="python evaluate_splicing.py {params.other_params} {params.with_PBD}  --vary_intronH_list {params.vary_intronH_list} --feat_idx_to_vary 1 "
        eval $command
        """
