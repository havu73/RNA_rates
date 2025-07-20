import numpy as np
import pandas as pd
from itertools import product
# there are 3 exons
# we will change the h for the second exon only for now
# the last exon will have a fixed h=1
h_range = [0.6,0.8,1, 1.2, 2.4, 3.6] # kb/minute
h_combinations = list(product(h_range, h_range, [1]))  #each h combination will be run as a job and will output one result_df
nExons= 3
length_tiers = [[0.5, 1, 1.5],
                [2, 3.5 ,5],
                [6,9,12],
                [15,22.5, 30]] # in kb
output_folder = 'results/without_PBD/no_fragment_read_filter/oneMinLabel/'
PBD = False

rule all:
    input:
        expand(os.path.join(output_folder, 'h1{h1}/h2{h2}/lengthTier{tier}.csv'), h1=h_range, h2=h_range, tier=[0])

def find_correct_tier(wildcards):
    return ' '.join(f'{f:.1f}' for f in length_tiers[int(wildcards.tier)])

rule run_evaluation:
    input:
    output:
        os.path.join(output_folder, 'h1{h1}/h2{h2}/lengthTier{tier}.csv'),
    params:
        exon_h_list = '{h1} {h2} 1',
        length_range = find_correct_tier,
        with_PBD = "--PBD" if PBD else ""
    shell:
        """
        echo {params.exon_h_list}
        echo {params.length_range}
        python evaluate_elongation_est.py --exon_h_list {params.exon_h_list} --length_range {params.length_range} --feat_idx_to_vary 0 1 --output_fn {output} {params.with_PBD} --time 1
        """