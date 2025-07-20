#!/bin/bash

# Parent directory containing directories with names starting with "5eUrates"
parent_dir="/pi/athma.pai-umw/analyses/jesse/KB/current_sci/simulations/reads_200/bysubrate"

count=0

# Iterate through directories starting with "5eUrates"
for dir in "$parent_dir"/5eUrates*; do
	if [ ! -d "$dir/split_mapped_reads/timepoints_splitted/readnames" ]; then
		echo "Directory '$dir' does not contain 'readnames' subdirectory."
		((count++)) #increment the counter
	fi
done

echo "Total directories without 'readnames': $count"
