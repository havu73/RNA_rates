'''
Given a gtf file of gene features that I inherited from Jesse, convert it to a format that I can actually use for simulation.
- gtf_df from Jesse is such that gene start and gene end is 1-based, include-include
- gtf_df for simulation is 0-based, include-exclude
'''
import pandas as pd
import numpy as np
import os
import sys
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Check continuous genes")
    parser.add_argument('--gtf_fn', type=str, help='input gene features')
    parser.add_argument('--output_fn', type=str, help='output simulation gene features')
    args = parser.parse_args()
    df = pd.read_csv(args.gtf_fn, sep='\t', header=0, index_col=None)
    df['start'] = df['start'] - 1
    # now add the PAS to the end of the gene, just 1bp downstream of the end of the gene
    PAS_line = {'chromosome': df.loc[0,'chromosome'],
                'source': df.loc[0, 'source'],
                'feature': 'PAS',
                'start': df.loc[df.shape[0]-1, 'end'],
                'end': df.loc[df.shape[0]-1, 'end']+ 1,
                'length': 1}  # chromosome      source  feature start   end     length  txrate  time    sequence
    df.loc[df.shape[0]] = PAS_line
    RTR_length = 1
    RTR_line = {'chromosome': df.loc[0,'chromosome'],
                'source': df.loc[0, 'source'],
                'feature': 'RTR',
                'start': df.loc[df.shape[0]-1, 'end'],
                'end': df.loc[df.shape[0]-1, 'end']+ RTR_length,
                'length': RTR_length}  # chromosome      source  feature start   end     length  txrate  time    sequence
    # because we do not have the sequence for the RTR, we will keep it minimal and simulate the elongtation with mostly just the genes' exons and intron
    df.loc[df.shape[0]] = RTR_line
    df['intron_h'] = 5
    df.drop(columns=['txrate', 'time'], inplace=True)
    df['is_intron'] = df['feature'].str.startswith('intron')
    df['is_PAS'] = df['feature'] == 'PAS'
    df['PAS_h'] = 1
    df.to_csv(args.output_fn, sep='\t', header=True, index=False, compression='gzip')
