code='create_design_matrix.py'
big_out_folder=/gladstone/engelhardt/lab/hvu/RNA_rates/elong_rates/true_gene_gtf/
mkdir -p ${big_out_folder}
rm -rf ${big_out_folder}/constant/*/*.gtf.gz
python ${code} --org_gtf_fn /gladstone/engelhardt/lab/hvu/RNA_rates/elong_rates/true_gene_gtf/gtf_files/ENSG00000000003_ENST00000612152.gtf.gz --time_traverse_gene 3 6 12 15 30 60  --output_folder ${big_out_folder}/constant/ --save_fn ${big_out_folder}/constant/design_matrix.csv  --pair_end  --insertsize_min 200 --insertsize_max 300 --read_length 100

#for fold in 2 4
#  do
#    for vari_bin_bp in 200 500 1000
#      do
#        bin_kb=$(awk "BEGIN {printf \"%.1f\", $vari_bin_bp / 1000}")
#        output_folder=${pair_end_folder}/bin${bin_kb}_fold${fold}/
#        echo $output_folder
#        python create_design_params.py --time_traverse_gene 3 6 12 15 30 60 --feat_length 1.65 3.35 6.75 13.5 27 --output_folder ${output_folder} --save_fn ${output_folder}/design_matrix.csv  --pair_end  --insertsize_min 200 --insertsize_max 300 --read_length 100 --vari_bin_bp ${vari_bin_bp} --vari_fold ${fold}
#      done
#  done
