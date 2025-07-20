#!/bin/bash

reads_1=$1
reads_2=$2
dir_out=$3
ref_genome=$4
nthreads=$5
file_name=$6

# Remove the file extension and path
#file_name=${reads_1##*/}
#file_name=${file_name%%.*}

# Create the name of the output files
file_out_sam_GA="$file_name"_GA.sam
file_out_bam_GA="$file_name"_GA.bam

# Run hisat3n to map the reads
#hisat-3n -x $ref_genome -q -1 $reads_1 -2 $reads_2 -S "$dir_out"/"$file_out_sam_GA" --base-change G,A -p 10 --rna-strandness RF
hisat-3n -x $ref_genome -q -1 $reads_1 -2 $reads_2 -S "$dir_out"/"$file_out_sam_GA" --base-change G,A -p $nthreads --rna-strandness RF --no-temp-splicesite --unique-only
#./hisat-3n -x '/gladstone/engelhardt/lab/hvu//data/ucsc/hg38/sequences/chrX' -q -1 ../2_chopper/2_chopperparameters_ENSG00000000003_150PE_R1.fastq.gz  -2 ../2_chopper/2_chopperparameters_ENSG00000000003_150PE_R2.fastq.gz -S ../3_hisat/ENSG00000000003_150PE_CT.sam --base-change C,T --rna-strandness RF --no-temp-splicesite --unique-only
#./hisat-3n -x '/gladstone/engelhardt/lab/hvu//data/ucsc/hg38/sequences/chrX' -q -1 ../2_chopper/2_chopperparameters_ENSG00000000003_150PE_R1.fastq.gz  -2 ../2_chopper/2_chopperparameters_ENSG00000000003_150PE_R2.fastq.gz -S ../3_hisat/ENSG00000000003_150PE_GA.sam --base-change G,A --rna-strandness RF --no-temp-splicesite --unique-only
# Convert the file to BAM format
samtools view -h -bS -F 260 "$dir_out"/"$file_out_sam_GA" | samtools sort > "$dir_out"/"$file_out_bam_GA"
#~/ucsc_bin/samtools-1.20/samtools view -h -bS -F 260 3_hisat/ENSG00000000003_150PE_CT.sam | ~/ucsc_bin/samtools-1.20/samtools sort > 4_bamToSam/ENSG00000000003_150PE_CT.bam
#~/ucsc_bin/samtools-1.20/samtools view -h -bS -F 260 3_hisat/ENSG00000000003_150PE_GA.sam | ~/ucsc_bin/samtools-1.20/samtools sort > 4_bamToSam/ENSG00000000003_150PE_GA.bam
rm "$dir_out"/"$file_out_sam_GA"
