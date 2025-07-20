#!/bin/bash
# to run -> bash /pi/athma.pai-umw/analyses/jesse/KB/current_sci/module_scripts/parameter_sorter.sh targetdirectory
# be sure to change the source and target directories accordingly!!

target_dir=$1
#target_dir="/pi/athma.pai-umw/analyses/jesse/KB/current_sci/simulations/reads_200"

export BSUB_QUIET=Y


# Iterate through XX, YY, and ZZ from 5 to 15
for ((XX=1; XX<=1; XX++)); do
    for ((YY=1; YY<=1; YY++)); do
        for ((ZZ=1; ZZ<=1; ZZ++)); do
            pattern="5eUrates_${XX}-${XX}_6sGrates_${YY}-${YY}_4sUrates_${ZZ}-${ZZ}_ptime_5_seqerr_0.1-0.2_reads_200.tsv"
	    #old pattern: "u2u${XX},${XX}_g2a${YY},${YY}_u2c${ZZ},${ZZ}_reads200.csv.gz"
            
	    new_dir="${target_dir}/5eUrates_${XX}-${XX}_6sGrates_${YY}-${YY}_4sUrates_${ZZ}-${ZZ}_ptime_5_seqerr_0.1-0.2_reads_200/mRNAs"
	    #old new_dir: "${target_dir}/u2u${XX},${XX}_g2a${YY},${YY}_u2c${ZZ},${ZZ}"
            
            # Create the new directory if it doesn't exist
            mkdir -p "${new_dir}"
            
            # Find files matching the pattern in sub-directories starting with 'ENSG' and copy them into directories that are named based on the substitution rates iterated in the simulations
            bsub -q short -n 5 -R rusage[mem=2000] -R span[hosts=1] -W 3:00 -J "$pattern" bash -c "find '${target_dir}' -type f -path '*/ENSG*/*${pattern}' -exec cp {} '${new_dir}/' \;"
        done
    done
done
