code='create_design_params.py'
pair_end_folder=/gladstone/engelhardt/lab/hvu/RNA_rates/elong_rates/pair_end_reads/variable/
#python create_design_params.py --time_traverse_gene 3 6 12 15 30 60 --feat_length 1.65 3.35 6.75 13.5 27 --output_folder ${pair_end_folder}/constant/ --save_fn ${pair_end_folder}/constant/design_matrix.csv  --pair_end  --insertsize_min 200 --insertsize_max 300 --read_length 100

for fold in 2 4
  do
    for vari_bin_bp in 200 500 1000
      do
        bin_kb=$(awk "BEGIN {printf \"%.1f\", $vari_bin_bp / 1000}")
        output_folder=${pair_end_folder}/bin${bin_kb}_fold${fold}/
        echo $output_folder
        python create_design_params.py --time_traverse_gene 3 6 12 15 30 60 --feat_length 1.65 3.35 6.75 13.5 27 --output_folder ${output_folder} --save_fn ${output_folder}/design_matrix.csv  --pair_end  --insertsize_min 200 --insertsize_max 300 --read_length 100 --vari_bin_bp ${vari_bin_bp} --vari_fold ${fold}
      done
  done
