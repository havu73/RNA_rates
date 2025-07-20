#!/bin/bash
# before running, conda activate kinetic_barcoding
# to run (until KB_master_wrapper.sh is ready to execute all modules) -> bash /pi/athma.pai-umw/analyses/jesse/KB/current_sci/module_scripts/mRNA_generator_caller.sh /pi/athma.pai-umw/analyses/jesse/KB/current_sci/simulations/reads_200 (outdir)

dir_in=/Users/hvu/PycharmProjects/RNA_rates/KB_Sims_gtfs
scriptname=/Users/hvu/PycharmProjects/RNA_rates/mRNA_generator.py
dir_out=$1
labeling_time=5
range_seqerr=0.1,0.2
min_5eU=1
max_5eU=1
min_6sG=1
max_6sG=1
min_4sU=1
max_4sU=1
num_reads=200

export BSUB_QUIET=Y

for i in "${dir_in}"/*.gtf
	do
	file_name=${i##*/}
	echo 'file_name: ' $file_name
	mkdir "${dir_out}"/"${file_name}"
		for j in $(seq $min_5eU $max_5eU)
			do
				for k in $(seq $min_6sG $max_6sG)
					do
						for l in $(seq $min_4sU $max_4sU)
							do
							
							#echo "${file_name}" #"${dir_out}"/"${file_name}"."${j}"."${k}"."${l}".e
							command="bsub -q long -n 1 -R rusage[mem=2000] -R span[hosts=1] -W 00:10 -J $file_name -o ${dir_out}/${file_name}/${file_name}.o -e ${dir_out}/${file_name}/${file_name}.e python $scriptname --f ${i} --l ${labeling_time} --b $range_seqerr --subs_5eU $j,$j --subs_6sG $k,$k --subs_4sU $l,$l --reads $num_reads --o ${dir_out}/${file_name}"
							echo -e $command
							done
					done
			done
	done

#for j in "$dir_out"/*.csv.gz
#	do 
#	gunzip -cd $j | sed 's|[[,]||g' | sed 's|[],]||g'| gzip > $j.mod.gz
#	done
