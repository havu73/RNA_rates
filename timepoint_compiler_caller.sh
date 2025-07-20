#!/bin/bash
# to run -> bash /pi/athma.pai-umw/analyses/jesse/KB/current_sci/module_scripts/timepoint_compiler_caller.sh target_dir

# example target_dir: /pi/athma.pai-umw/analyses/jesse/KB/current_sci/simulations/reads_200/bysubrate

module add samtools/1.16.1

target_dir=$1
scriptname=/pi/athma.pai-umw/analyses/jesse/KB/current_sci/module_scripts/timepoint_compiler.sh

export BSUB_QUIET=Y

# Iterate through XX, YY, and ZZ from 5 to 15
for ((XX=1; XX<=1; XX++)); do
	for ((YY=1; YY<=1; YY++)); do
		for ((ZZ=1; ZZ<=1; ZZ++)); do
                
			dir="5eUrates_${XX}-${XX}_6sGrates_${YY}-${YY}_4sUrates_${ZZ}-${ZZ}_ptime_5_seqerr_0.1-0.2_reads_200"
			dir_in="${target_dir}"/"${dir}"/split_mapped_reads/timepoints_splitted
			dir_out="${target_dir}"/"${dir}"/split_mapped_reads/timepoints_splitted/readnames

			mkdir -p "${dir_out}"
				
			for file in "${dir_in}"/*.bam; do
			
				file_name=${file##*/}
				file_name=${file_name%%.bam}
                
				#echo the file with path is "$file"
				#echo the file name after trimming is "$file_name"

				#echo the output and error files will be named "$dir_out"/"$file_name"

				# Run the timepoint_compiler.sh script
				bsub -q short -n 1 -R rusage[mem=30000] -R span[hosts=1] -J "${file_name}" -o "${dir_out}"/"${file_name}".o -e "${dir_out}"/"${file_name}".e -W 1:00 $scriptname $file_name $dir_in $dir_out

			done
		done
	done
done
