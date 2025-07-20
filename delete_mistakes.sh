#!/bin/bash
# to run -> bash /pi/athma.pai-umw/analyses/jesse/KB/current_sci/module_scripts/delete_mistakes.sh

# Parent directory containing directories with names starting with "5eUrates"
parent_dir="/pi/athma.pai-umw/analyses/jesse/KB/current_sci/simulations/reads_200/bysubrate"

# Iterate through directories starting with "5eUrates"
for dir in "$parent_dir"/5eUrates*; do
    # Check if the subdirectories exist before attempting to delete them
    if [ -d "$dir/split_mapped_reads/timepoints_splitted/readnames" ]; then
        rm -r "$dir/split_mapped_reads/timepoints_splitted/readnames"
        #echo "Deleted 'mapped_reads' directory in '$dir'."
    fi

    #if [ -d "$dir/split_mapped_reads" ]; then
        #rm -r "$dir/split_mapped_reads"
        #echo "Deleted 'split_mapped_reads' directory in '$dir'."
    #fi
done

