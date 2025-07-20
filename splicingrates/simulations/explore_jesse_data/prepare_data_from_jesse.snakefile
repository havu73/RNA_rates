import pandas as pd
import os
import glob
all_folder = '/gladstone/engelhardt/lab/hvu/RNA_rates/data_from_jesse/'
raw_data_folder = os.path.join(all_folder, 'pi/athma.pai-umw/analyses/jesse/KB/variable_elongation_rates/region_100-5000_reads_300_iter_100/region_specific_elongation_rate_gtfs')  # this is the folder that appears if I unzip the file region_100-5000_reads_300_iter_100_ground_truth_gtfs.tar.gz
gtf_folder = os.path.join(all_folder, 'gtf_ground_truth')
coverage_folder= os.path.join(all_folder, 'coverage')
ha_folder='/gladstone/engelhardt/lab/hvu/RNA_rates/data_from_jesse/ha_reorganized_data'
iter_list = range(1, 101)

example_gtf_folder='/gladstone/engelhardt/lab/hvu/RNA_rates/data_from_jesse/gtf_ground_truth/iter_33'
gtf_fn_list = glob.glob(os.path.join(example_gtf_folder, '*.gtf.gz'))
gtf_fn_list = list(map(lambda x: os.path.basename(x), gtf_fn_list))
gene_name_list = list(map(lambda x: os.path.basename(x).split('.')[0].split('_')[0], gtf_fn_list))
gene_name_list = list(map(lambda x: x+'/'+x, gene_name_list))
print(gene_name_list[:3])
rule all:
    input:
        #expand(os.path.join(gtf_folder, 'iter_{iter}', '{gtf_fn}') , iter=iter_list, gtf_fn=gtf_fn_list),
        expand(os.path.join(ha_folder, 'iter_{iter}', 'ENSG00000005812/ENSG00000005812.gtf.gz'), iter=iter_list),
        expand(os.path.join(ha_folder, 'iter_{iter}', 'ENSG00000005812/ENSG00000005812_nt_coverage_0-5.bed.gz'), iter=iter_list),
        expand(os.path.join(ha_folder, 'iter_{iter}', 'ENSG00000005812/ENSG00000005812_nt_coverage_5-10.bed.gz'), iter=iter_list),
        expand(os.path.join(ha_folder, 'iter_{iter}', 'ENSG00000005812/ENSG00000005812_nt_coverage_10-15.bed.gz'), iter=iter_list)

rule move_gtf_to_folder:
    input:
    output:
        expand(os.path.join(gtf_folder, 'iter_{{iter}}', '{gtf_fn}'), gtf_fn=gtf_fn_list),
    params:
        output_folder=os.path.join(gtf_folder, 'iter_{iter}')
    shell:
        """
        mkdir -p {params.output_folder}
        mv {raw_data_folder}/iter_{wildcards.iter}/*.gtf {params.output_folder}
        gzip {params.output_folder}/*.gtf
        """


rule link_gtf_to_gene_folder:
    input:
        expand(os.path.join(gtf_folder, 'iter_{{iter}}', '{gtf_fn}'), gtf_fn=gtf_fn_list)
    output:
        os.path.join(ha_folder, 'iter_{iter}', 'ENSG00000005812/ENSG00000005812.gtf.gz')
    params:
        org_folder=os.path.join(gtf_folder, 'iter_{iter}'),
        dest_folder=os.path.join(ha_folder, 'iter_{iter}')
    shell:
        """
        mkdir -p {params.dest_folder}
        for f in {params.org_folder}/*ntRatesandTraversalTimes.gtf.gz; do
            gene_name=$(basename $f | cut -d'_' -f1)
            mkdir -p {params.dest_folder}/$gene_name
            ln -s $f {params.dest_folder}/$gene_name/$gene_name.gtf.gz
        done
        """



rule link_coverage_to_gene_folder:
    input:
        os.path.join(coverage_folder, 'iter_{{iter}}', 'ENSG00000005812_nt_coverage_0-5.bed.gz'),
        os.path.join(coverage_folder, 'iter_{{iter}}', 'ENSG00000005812_nt_coverage_5-10.bed.gz'),
        os.path.join(coverage_folder, 'iter_{{iter}}', 'ENSG00000005812_nt_coverage_10-15.bed.gz'),
    output:
        os.path.join(ha_folder, 'iter_{iter}', 'ENSG00000005812/ENSG00000005812_nt_coverage_0-5.bed.gz'),
        os.path.join(ha_folder, 'iter_{iter}', 'ENSG00000005812/ENSG00000005812_nt_coverage_5-10.bed.gz'),
        os.path.join(ha_folder, 'iter_{iter}', 'ENSG00000005812/ENSG00000005812_nt_coverage_10-15.bed.gz')
    params:
        org_folder=os.path.join(coverage_folder, 'iter_{iter}'),
        dest_folder=os.path.join(ha_folder, 'iter_{iter}'),
    shell:
        """
        mkdir -p {params.dest_folder}
        for f in {params.org_folder}/*nt_coverage_0-5.bed.gz; do
            gene_name=$(basename $f | cut -d'_' -f1)
            mkdir -p {params.dest_folder}/$gene_name
            ln -s {params.org_folder}/${gene_name}_nt_coverage_0-5.bed.gz {params.dest_folder}/$gene_name/
            ln -s {params.org_folder}/${gene_name}_nt_coverage_5-10.bed.gz {params.dest_folder}/$gene_name/
            ln -s {params.org_folder}/${gene_name}_nt_coverage_10-15.bed.gz {params.dest_folder}/$gene_name/
        done
        """