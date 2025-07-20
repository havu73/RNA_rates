#!/bin/bash

dir_in=$1
dir_out=$2
dir_temp=$3
name=$4
nthreads=$5
dir_individual_timepoints=$6
dir_timepoints_cumulative=$7


#Get headers
# not sure if this is even needed , maybe in the script of timepoint_classifier.R
samtools view -H "$dir_in"/"$name"_GA.bam > "$dir_temp"/"$name"_GA.sam.header
samtools view -H "$dir_in"/"$name"_TC.bam > "$dir_temp"/"$name"_TC.sam.header

#Get reads names with T>C subs
samtools view -F 4 -@ $nthreads -D Yf:/pi/athma.pai-umw/analyses/jesse/KB/current_sci/essential_simulation_files/tag_values_shortened.txt "$dir_in"/"$name"_TC.bam | cut -f1 | sort | uniq > "$dir_temp"/"$name"_TC_read_names.txt

# samtools view -F 4 : exclude reads that are unmapped
# cut -f1 : get the first column of the output
#Get reads names with G>A subs
samtools view -F 4 -@ $nthreads -D Yf:/pi/athma.pai-umw/analyses/jesse/KB/current_sci/essential_simulation_files/tag_values_shortened.txt "$dir_in"/"$name"_GA.bam | cut -f1 | sort | uniq > "$dir_temp"/"$name"_GA_read_names_prelim.txt

#Remove from G>A reads the ones that also have T>Cs
# comm -23 : Compare the sort files fn1 and fn2 and get only lines that appear exclusively in fn1
# the flag -23 suppresses the lines that are common to both files, and lines that are in the second file only
comm -23 "$dir_temp"/"$name"_GA_read_names_prelim.txt "$dir_temp"/"$name"_TC_read_names.txt > "$dir_temp"/"$name"_GA_read_names.txt

#Get a list of labeled reads (either G>A or T>C) --> total read names
cat "$dir_temp"/"$name"_GA_read_names.txt "$dir_temp"/"$name"_TC_read_names.txt | sort | uniq > "$dir_temp"/"$name"_labeled_read_names.txt

#Now get the reads using picard
java -jar /pi/athma.pai-umw/analyses/jesse/KB/current_sci/module_scripts/picard.jar FilterSamReads -I "$dir_in"/"$name"_TC.bam -O "$dir_individual_timepoints"/"$name"_10-15.bam -READ_LIST_FILE "$dir_temp"/"$name"_TC_read_names.txt -FILTER includeReadList -SORT_ORDER coordinate
# the above command will get the reads in the original bam files (which were aligned with GA as substitutions). The point is, with the same set of reads and 2 runs of hisat (CT and GA), we separate the reads that are
java -jar /pi/athma.pai-umw/analyses/jesse/KB/current_sci/module_scripts/picard.jar FilterSamReads -I "$dir_in"/"$name"_GA.bam -O "$dir_individual_timepoints"/"$name"_5-10.bam -READ_LIST_FILE "$dir_temp"/"$name"_GA_read_names.txt -FILTER includeReadList -SORT_ORDER coordinate
java -jar /pi/athma.pai-umw/analyses/jesse/KB/current_sci/module_scripts/picard.jar FilterSamReads -I "$dir_in"/"$name"_GA.bam -O "$dir_individual_timepoints"/"$name"_0-5.bam -READ_LIST_FILE "$dir_temp"/"$name"_labeled_read_names.txt -FILTER excludeReadList -SORT_ORDER coordinate
#Because the reads with no substitutions are assigned to 0-5, it doesn't matter from which bam file I pull those reads, they won't have substitutions in any of the bams.

#Generate cumulative bam files
cp "$dir_individual_timepoints"/"$name"_0-5.bam "$dir_timepoints_cumulative"/"$name"_5eU.bam
# samtools merge [options] out.bam in1.bam ... inN.bam
samtools merge -@ $nthreads "$dir_timepoints_cumulative"/"$name"_5eU+6sG.bam "$dir_individual_timepoints"/"$name"_0-5.bam "$dir_individual_timepoints"/"$name"_5-10.bam
samtools merge -@ $nthreads "$dir_timepoints_cumulative"/"$name"_5eU+6sG+4sU.bam "$dir_timepoints_cumulative"/"$name"_5eU+6sG.bam "$dir_individual_timepoints"/"$name"_10-15.bam

#Index all
samtools index "$dir_individual_timepoints"/"$name"_0-5.bam
samtools index "$dir_individual_timepoints"/"$name"_5-10.bam
samtools index "$dir_individual_timepoints"/"$name"_10-15.bam

samtools index "$dir_timepoints_cumulative"/"$name"_5eU.bam
samtools index "$dir_timepoints_cumulative"/"$name"_5eU+6sG.bam
samtools index "$dir_timepoints_cumulative"/"$name"_5eU+6sG+4sU.bam

#Remove temp files
rm "$dir_temp"/"$name"*



