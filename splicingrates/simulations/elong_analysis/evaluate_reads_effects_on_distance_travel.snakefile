'''
The purpose of this snakefile is to create an analysis to show that even when we try to calculate the distance travelled using only reads and without the steps of inferring the transcript length from the 
reads, we can still achieve good accuracy in terms of the distance travel within m minutes of the kinetic barcoding experiment
'''
import numpy as np
import pandas as pd
from itertools import product

big_folder = '/gladstone/engelhardt/lab/hvu/RNA_rates/elong_rates/'
output_folder = '/gladstone/engelhardt/lab/hvu/RNA_rates/elong_rates/analyse_x0_x1/'
subfolder_list = ['constant']
design_fn = '/gladstone/engelhardt/lab/hvu/RNA_rates/elong_rates/no_reads/constant/design_matrix.csv'
def num_lines_in_file(fn):
    df = pd.read_csv(fn, header =0, index_col=None, sep = '\t')
    return df.shape[0]

num_jobs = num_lines_in_file(design_fn)

jobs_to_care = list(range(num_jobs))
jobs_to_care.remove(25)
print(jobs_to_care)
rule all:
    input:
        expand(os.path.join(output_folder,  '{subfolder}', 'result_{job_idx}_x0_x1.csv.gz'), job_idx = jobs_to_care, subfolder = subfolder_list)

rule calculate_x0_x1:
    input:
        os.path.join(big_folder,  'pair_end_reads', '{subfolder}', 'result_{job_idx}', 'x0_x1.csv.gz'),
        os.path.join(big_folder,  'no_reads', '{subfolder}', 'result_{job_idx}', 'x0_x1.csv.gz'),
        os.path.join(big_folder,  'no_reads', '{subfolder}', 'result_{job_idx}', 'input_elongf_df.csv')
    output:
        os.path.join(output_folder,  '{subfolder}', 'result_{job_idx}_x0_x1.csv.gz')
    shell:
        '''
        command="python analysis_x0_x1_with_reads.py --with_reads_fn {input[0]} --no_reads_fn {input[1]} --elong_fn {input[2]} --output_fn {output}"
        echo $command
        eval $command
        '''
