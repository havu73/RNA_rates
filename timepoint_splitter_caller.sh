#!/bin/bash
# to run --> bash /pi/athma.pai-umw/analyses/jesse/KB/current_sci/module_scripts/timepoint_splitter_caller.sh target_dir

# example target_dir: /pi/athma.pai-umw/analyses/jesse/KB/current_sci/simulations/reads_200/bysubrate

# This script is meant to be used after hisat3n for G>As and T>Cs, so it needs a pair of bam files (one per substitution type)

module add samtools/1.16.1
module add picard/2.27.5
module add bedtools/2.30.0

target_dir=$1
nthreads=2
scriptname=/pi/athma.pai-umw/analyses/jesse/KB/current_sci/module_scripts/timepoint_splitter.sh 

export BSUB_QUIET=Y

# Iterate through XX, YY, and ZZ from 5 to 15
for ((XX=1; XX<=1; XX++)); do
	for ((YY=1; YY<=1; YY++)); do
		for ((ZZ=1; ZZ<=1; ZZ++)); do
                	
			# store the pattern for identifying the directories within the "bysubrate" parent directory
        		dir="5eUrates_${XX}-${XX}_6sGrates_${YY}-${YY}_4sUrates_${ZZ}-${ZZ}_ptime_5_seqerr_0.1-0.2_reads_200"
            
           		# store the directories containing the hisat-3n mapped reads
			dir_in="$target_dir"/"$dir"/mapped_reads
					
			# make and store the output directory
			dir_out="$target_dir"/"$dir"/split_mapped_reads
			mkdir "$dir_out"
					
			# make and store various subdirectories within the output directory
			mkdir "$dir_out"/temp
			dir_temp="$dir_out"/temp

			mkdir "$dir_out"/timepoints_splitted
			dir_individual_timepoints="$dir_out"/timepoints_splitted

			mkdir "$dir_out"/timepoints_cumulative
			dir_timepoints_cumulative="$dir_out"/timepoints_cumulative

			#echo the pattern is "$dir"
			#echo the input directory is "$dir_in"
			#echo the output directory is "$dir_out"

			for gene in ENSG00000000003 ENSG00000000419 ENSG00000000460 ENSG00000001036 ENSG00000001084 ENSG00000001497 ENSG00000001629 ENSG00000001630 ENSG00000001631 ENSG00000002016 ENSG00000002330 ENSG00000002586 ENSG00000002822 ENSG00000002834 ENSG00000002919 ENSG00000003056 ENSG00000003249 ENSG00000003393 ENSG00000003400 ENSG00000003402 ENSG00000003436 ENSG00000003509 ENSG00000003756 ENSG00000003989 ENSG00000004059 ENSG00000004142 ENSG00000004399 ENSG00000004478 ENSG00000004487 ENSG00000004534 ENSG00000004700 ENSG00000004766 ENSG00000004779 ENSG00000004864 ENSG00000004866 ENSG00000004897 ENSG00000004961 ENSG00000004975 ENSG00000005007 ENSG00000005020 ENSG00000005022 ENSG00000005075 ENSG00000005100 ENSG00000005156 ENSG00000005189 ENSG00000005194 ENSG00000005339 ENSG00000005436 ENSG00000005448 ENSG00000005483 ENSG00000005486 ENSG00000005700 ENSG00000005801 ENSG00000005810 ENSG00000005812 ENSG00000005889 ENSG00000005893 ENSG00000006015 ENSG00000006016 ENSG00000006042 ENSG00000006194 ENSG00000006210 ENSG00000006377 ENSG00000006451 ENSG00000006453 ENSG00000006459 ENSG00000006576 ENSG00000006607 ENSG00000006625 ENSG00000006695 ENSG00000006744 ENSG00000006756 ENSG00000006831 ENSG00000007202 ENSG00000008018 ENSG00000008256 ENSG00000008294 ENSG00000059691 ENSG00000059769 ENSG00000064726 ENSG00000066697 ENSG00000068724 ENSG00000071082 ENSG00000073417 ENSG00000074842 ENSG00000075413 ENSG00000078237 ENSG00000080815 ENSG00000080822 ENSG00000080823 ENSG00000086200 ENSG00000110955
				do
				
				# store the name of the file to be used in insert script
				file_name="$dir"_"$gene"
				
				# store the file name with its directory
				file="$dir_in"/"$file_name"
           		
           			#echo the gene for this test is "$gene"
           			#echo the root of the file name for this test is "$file_name"
           			#echo the location of this file is "$file"
           		
				bsub -q long -n $nthreads -R rusage[mem=30000] -J $file_name -o "$dir_out"/"$file_name".o -e "$dir_out"/"$file_name".e -W 5:00 $scriptname $dir_in $dir_out $dir_temp $file_name $nthreads $dir_individual_timepoints $dir_timepoints_cumulative
			done
		done
	done
done



