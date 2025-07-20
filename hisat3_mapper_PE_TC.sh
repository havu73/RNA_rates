#!/bin/bash

reads_1=$1
reads_2=$2
dir_out=$3 # /gladstone/engelhardt/home/hvu/source/RNA_rates/3_mapped_reads
ref_genome=$4
nthreads=$5
file_name=$6

# Remove the file extension and path
#file_name=${reads_1##*/}
#file_name=${file_name%%.*}

# Create the name of the output files
file_out_sam_TC="$file_name"_TC.sam
file_out_bam_TC="$file_name"_TC.bam

# Run hisat3n to map the reads
#/gladstone/engelhardt/home/hvu/source/RNA_rates/hisat-3n/hisat-3n
hisat-3n -x $ref_genome -q -1 $reads_1 -2 $reads_2 -S "$dir_out"/"$file_out_sam_TC" --base-change T,C -p $nthreads --rna-strandness RF --no-temp-splicesite --unique-only

# Convert the file to BAM format
samtools view -h -bS -F 260 "$dir_out"/"$file_out_sam_TC" | samtools sort > "$dir_out"/"$file_out_bam_TC"

#Index
samtools index "$dir_out"/"$file_out_bam_TC"

rm "$dir_out"/"$file_out_sam_TC"
