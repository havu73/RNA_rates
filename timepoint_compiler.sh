#!/bin/bash

file_name=$1
dir_in=$2
dir_out=$3

samtools view "${dir_in}"/"${file_name}".bam | cut -f 1 | awk '!x[$0]++' > "${dir_out}"/"${file_name}".readnames.txt


# the command will take in the read names (first column) and remove duplicates (preserving order)
#The command awk '!x[$0]++' filters out duplicate lines from the input, printing only the first occurrence of each line. It preserves the order of the lines as they first appear in the input.