import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import helper
import pysam
TOSS_DISTANCE = 300  # usually in simulation, we do not care about reads that are among the first TOSS_DISTANCE FIRST bases of the gene. We will not consider them in our analysis
def calculate_coverage_from_sam(sam_file, output_folder, refseq_name):
    samfile = pysam.AlignmentFile(sam_file, "r")
    start = TOSS_DISTANCE  # start position
    end = samfile.get_reference_length(refseq_name)  # end position
    coverage = samfile.count_coverage(contig=refseq_name, start=start, end=end)
    # coverage is a tuple of four arrays (A, C, G, T coverage)
    # To get total coverage at each position, sum across the four arrays
    total_coverage = coverage[0] + coverage[1] + coverage[2] + coverage[3]
    samfile.close()
    return total_coverage

if __name__== '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sam_file', required=True, help='Input .sam file recording the reads and their coordinates')
    parser.add_argument('--output_folder', required=True, help='output_folder')
    parser.add_argument('--refseq_name', required=True, help='refseq_name. for example, if we generate reads around an intron as part of simulation. Here, we would like to calculate the coverage of each genomic position alogn that intron and surrounding it. This parameter will let the code know that we will focus on calculating the coverage for that speciifc region')
    args = parser.parse_args()
    helper.check_file_exist(args.sam_file)
    helper.make_dir(args.output_folder)
    print("Done getting command line arguments")
    coverage = calculate_coverage_from_sam(args.sam_file, args.output_folder, args.refseq_name)
    print("Done calculating coverage from sam file")

"""
If I want to calculate read depth on a particular region of the reference genome using a .sam file and samtools I have to do the following steps:
- samtools view -bS -o input.sam > output.bam
- samtools sort output.bam > output.sorted.bam
- samtools index output.sorted.bam
- samtools depth -r chr1:1000-2000 output.sorted.bam > output.depth
Therefore, it may make more sense to just use the pysam package to calculate the read depth.
"""
