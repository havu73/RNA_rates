#!/bin/bash
# to run -> bash /pi/athma.pai-umw/analyses/jesse/KB/current_sci/module_scripts/chopper_R_caller.sh /pi/athma.pai-umw/analyses/jesse/KB/current_sci/simulations/reads_200/bysubrate

module add r/4.2.2

target_dir=$1
scriptname=/pi/athma.pai-umw/analyses/jesse/KB/current_sci/module_scripts/chopper.R
refbed=/pi/athma.pai-umw/analyses/jesse/KB/current_sci/essential_simulation_files/Homo_sapiens.GRCh38.95.uniquegene.bed

export BSUB_QUIET=Y

# Iterate through XX, YY, and ZZ from 5 to 15
for ((XX=1; XX<=1; XX++)); do
	for ((YY=1; YY<=1; YY++)); do
		for ((ZZ=1; ZZ<=1; ZZ++)); do
			
			dir="5eUrates_${XX}-${XX}_6sGrates_${YY}-${YY}_4sUrates_${ZZ}-${ZZ}_ptime_5_seqerr_0.1-0.2_reads_200"
			new_dir="${target_dir}"/"${dir}"/chopper
			
			mkdir -p "${new_dir}"
			
			#echo "$dir"
			#echo "$new_dir"
			#echo "$target_dir"/"$dir"
			
			bsub -q short -n 1 -R rusage[mem=10000] -R span[hosts=1] -W 1:00 -J ${dir}kb_chopper -o ${new_dir}/kb_chopper.o -e ${new_dir}/kb_chopper.e Rscript $scriptname $target_dir/$dir/mRNAs $refbed $target_dir/$dir/chopper/ $dir
		done
	done
done
