import pandas as pd
import numpy as np
import glob

def get_gene_list(large_folder):
    fn_list = glob.glob(large_folder + 'iter_1/*/*gtf.gz')
    gene_list = list(map(lambda x: x.split('/')[-2], fn_list))
    return gene_list

if __name__ == '__main__':
    large_folder = '/gladstone/engelhardt/lab/hvu/RNA_rates/data_from_jesse/ha_reorganized_data/'
    '''
    large_folder
    |__ iter_1
    |   |__ gene
    |       |__ gene.gtf.gz
    '''
    gene_list = get_gene_list(large_folder)
    df = pd.DataFrame(gene_list, columns=['gene'])
    df['num_iteration_done'] = df['gene'].apply(lambda x: len(glob.glob(f'{large_folder}/iter_*/{x}/pred_h.csv.gz')))  # count the number of iterations for which this gene has
    df.to_csv('/gladstone/engelhardt/lab/hvu/RNA_rates/data_from_jesse/ha_reorganized_data/num_iterations_done.csv', index=False, sep='\t')

