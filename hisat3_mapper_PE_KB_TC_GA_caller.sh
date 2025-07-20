#!/bin/bash
# to run --> bash /pi/athma.pai-umw/analyses/jesse/KB/current_sci/module_scripts/hisat3_mapper_PE_KB_TC_GA_caller.sh target_dir

# target dir example: /pi/athma.pai-umw/analyses/jesse/KB/current_sci/simulations/reads_200/bysubrate

module add hisat-3n/2.2.1-3n-0.0.3
module add samtools/1.16.1

# edited from ECR script: TC and GA scriptname filepaths, reference genome, ENSG gene IDs in for-loop, reads 1 and 2 filename structure

scriptname_TC=/pi/athma.pai-umw/analyses/jesse/KB/current_sci/module_scripts/hisat3_mapper_PE_TC.sh
scriptname_GA=/pi/athma.pai-umw/analyses/jesse/KB/current_sci/module_scripts/hisat3_mapper_PE_GA.sh

export BSUB_QUIET=Y

target_dir=$1
ref_genome=/pi/athma.pai-umw/genomes/hg38/hisat-3n_HG38_WITHSS_ERCCs/hg38
nthreads=1

# Iterate through XX, YY, and ZZ from 5 to 15
for ((XX=1; XX<=1; XX++)); do
	for ((YY=1; YY<=1; YY++)); do
		for ((ZZ=1; ZZ<=1; ZZ++)); do
                	
			dir="5eUrates_${XX}-${XX}_6sGrates_${YY}-${YY}_4sUrates_${ZZ}-${ZZ}_ptime_5_seqerr_0.1-0.2_reads_200"
			source_dir="${target_dir}"/"${dir}"/chopper
			new_dir="${target_dir}"/"${dir}"/mapped_reads
                	
			mkdir -p "${new_dir}"
                	
			#echo the pattern is "$dir"
			#echo the output directory is "$new_dir"
			#echo the directory with input files is "$source_dir"
                        
			for file in ENSG00000000003 ENSG00000000419 ENSG00000000460 ENSG00000001036 ENSG00000001084 ENSG00000001497 ENSG00000001629 ENSG00000001630 ENSG00000001631 ENSG00000002016 ENSG00000002330 ENSG00000002586 ENSG00000002822 ENSG00000002834 ENSG00000002919 ENSG00000003056 ENSG00000003249 ENSG00000003393 ENSG00000003400 ENSG00000003402 ENSG00000003436 ENSG00000003509 ENSG00000003756 ENSG00000003989 ENSG00000004059 ENSG00000004142 ENSG00000004399 ENSG00000004478 ENSG00000004487 ENSG00000004534 ENSG00000004700 ENSG00000004766 ENSG00000004779 ENSG00000004864 ENSG00000004866 ENSG00000004897 ENSG00000004961 ENSG00000004975 ENSG00000005007 ENSG00000005020 ENSG00000005022 ENSG00000005075 ENSG00000005100 ENSG00000005156 ENSG00000005189 ENSG00000005194 ENSG00000005339 ENSG00000005436 ENSG00000005448 ENSG00000005483 ENSG00000005486 ENSG00000005700 ENSG00000005801 ENSG00000005810 ENSG00000005812 ENSG00000005889 ENSG00000005893 ENSG00000006015 ENSG00000006016 ENSG00000006042 ENSG00000006194 ENSG00000006210 ENSG00000006377 ENSG00000006451 ENSG00000006453 ENSG00000006459 ENSG00000006576 ENSG00000006607 ENSG00000006625 ENSG00000006695 ENSG00000006744 ENSG00000006756 ENSG00000006831 ENSG00000007202 ENSG00000008018 ENSG00000008256 ENSG00000008294 ENSG00000059691 ENSG00000059769 ENSG00000064726 ENSG00000066697 ENSG00000068724 ENSG00000071082 ENSG00000073417 ENSG00000074842 ENSG00000075413 ENSG00000078237 ENSG00000080815 ENSG00000080822 ENSG00000080823 ENSG00000086200 ENSG00000110955
				do
					file_name="$dir"_"$file"

					reads_1="$source_dir"/"$dir"_"$file"_150PE_R1.fastq.gz
					reads_2="$source_dir"/"$dir"_"$file"_150PE_R2.fastq.gz
            				
					#echo the pattern for read 1 is "$reads_1"
					#echo the pattern for read 2 is "$reads_2"

					#echo the output and error files will be named "$new_dir"/"$file_name"

					# Run hisat3n. HISAT3N is only available to multithread if the cores are on the same system!
					bsub -q long -n $nthreads -R rusage[mem=25000] -R span[hosts=1] -J $file -o "$new_dir"/"$file_name".o -e "$new_dir"/"$file_name".e -W 1:00 $scriptname_TC $reads_1 $reads_2 $new_dir $ref_genome $nthreads $file_name
					bsub -q long -n $nthreads -R rusage[mem=25000] -R span[hosts=1] -J $file -o "$new_dir"/"$file_name".o -e "$new_dir"/"$file_name".e -W 1:00 $scriptname_GA $reads_1 $reads_2 $new_dir $ref_genome $nthreads $file_name
			done
		done
	done
done

