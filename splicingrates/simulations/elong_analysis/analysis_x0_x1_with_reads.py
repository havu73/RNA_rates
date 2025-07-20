'''
The purpose of this script is to prove that even when we only have reads and not the full transcripts, we can still estimate elongation rates.
'''
import pandas as pd
import numpy as np
import argparse
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) # this is supposed to let the code know that some of the import path is outside of this folder
from transcription.elongation_calculation import calculate_enlongated_endsite
def read_x0_x1(fn, with_reads=True):
    x0_x1 = pd.read_csv(fn, sep='\t', header=0, index_col=None)
    if with_reads:
        x0_x1.columns = ['x0', 'paired_end_x1']
    else:
        x0_x1.columns = ['x0', 'no_reads_x1']
    return x0_x1

def compare_elongation_rates(with_reads_fn, no_reads_fn, elong_fn, time, output):
    elongf_df = pd.read_csv(elong_fn, sep='\t', header=0, index_col=None)
    with_reads = read_x0_x1(with_reads_fn, with_reads=True)
    no_reads = read_x0_x1(no_reads_fn, with_reads=False)
    merged = pd.merge(with_reads, no_reads, on='x0')
    merged['true_x1'] = merged['x0'].apply(lambda x: calculate_enlongated_endsite(x, elongf_df, time))
    merged['true_distance'] = merged['true_x1'] - merged['x0']
    merged['distance_with_reads'] = merged['paired_end_x1'] - merged['x0']
    merged['distance_no_reads'] = merged['no_reads_x1'] - merged['x0']
    merged.to_csv(output, sep='\t', index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('This code will try to take in the file x0_x1 from a simulation with reads and one without reads, and compare the distance travelled with the true distrance travelled (based on elongf_df)')
    parser.add_argument('--with_reads_fn', type=str, help='x0_x1 file in simulation with reads')
    parser.add_argument('--no_reads_fn', type=str, help='x0_x1 file in simulation without reads')
    parser.add_argument('--elong_fn', type=str, help='elongation rate file')
    parser.add_argument('--time', type=int, default=5, help='How long is the time between each kinetic barcoding experiment')
    parser.add_argument('--output_fn', type=str, help='output file')
    args = parser.parse_args()
    compare_elongation_rates(args.with_reads_fn, args.no_reads_fn, args.elong_fn, args.time, args.output_fn)


