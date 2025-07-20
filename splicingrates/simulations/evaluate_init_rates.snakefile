import numpy as np
import pandas as pd
from itertools import product
# there are 3 exons
# we will change the h for the second exon only for now
# the last exon will have a fixed h=1

output_folder = 'init_est/without_PBD'
PBD = False
time=5
lambda_init = [0.5, 1, 1.5, 2] + list(range(3, 5)) + list(range(5, 15, 3))
burst_size = list(range(5,20,5)) + list(range(20, 50, 10))
wiggle_room = [0.3]
# all the combination of lambda_init, burst_size, wiggle_room
params_comb = list(product(lambda_init, burst_size, wiggle_room))
comb_idx_list = list(range(len(params_comb)))

rule all:
    input:
        expand(os.path.join(output_folder, 'result_l{l}_b{b}.csv'), l=lambda_init, b=burst_size)


rule run_evaluation:
    input:
    output:
        os.path.join(output_folder, 'result_l{l}_b{b}.csv'),
    params:
        with_PBD = "--PBD" if PBD else "",
    shell:
        """
        python evaluate_init_est.py --output_fn {output} {params.with_PBD} --time {time} --lambda_init {wildcards.l} --burst_size {wildcards.b} --wiggle_room 0.3
        """