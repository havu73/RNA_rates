# this code is supposed to help debug some of the issues with the system to do piece-wise linear regression and calculation og elongation rates given the data of the read coverage
# the code is not tested, it is very ad-hoc and will need refinement but it is not a priority at the moment
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import unittest
import pandas as pd
import visualize_simulations as viz
import transcription.simulate_multiple_experiments as sim
import transcription.helper as helper
from estimates.elongation import estElong
from pandas.testing import assert_series_equal
from estimates.utils import merge_intervals, drop_trailing_zeros
target_exp = 5
num_total_transcript_millions = 100
ONE_KB=1000
degrade_rate = 0.00
intron_h = 5
PAS_h=5
RTR=500
lambda_init = 2 # average number of burst events per minute (burst event is the event where a burst_size transcripts are created around the same time)
burst_size = 10 # number of transcripts created in a burst event
wiggle_room = 0.3 # the wiggle room for the burst event. See the comments in function Experiment.init_bursting_transcripts to understand the meaning of this parameter. If not sure, don't modify it.
# if I want to simulate situation such that there is no read being generated from fragments (only the whole transcripts are sequenced), the following parameters should be set carefully:
eta_val=helper.DFT_ETA_VALUE  # the scale fo weibull distribution for the fragment length
insertsize_min = -1 #200  #filter fragments minimum length
insertsize_max = -1 #300  # filter fragments maximum length
read_length = -1 #150 # the length of the reads
frag_func= 'weibull_edge' # whether we will simulate the fragmentation of transcripts based on the weibull distribution or the uniform fragmentation method
unif_avg_frag_len=250 # the desired average fragment length if we use the uniform fragmentation method
# if I set the read values to -1, the program will just generate fragments and not get rid of any portion of the fragments.
simulate_cleavage=False ## for this problem, we don't need to simulate cleavage because we really only care about calculating the elongation speed of the transcripts. We skip splicing and cleavage for now.
PDB = False  # whether we simulate a system where there is no existing transcripts, and instead we used PDB to stop the transcripts from elongation, and let the elongation to start PDB_time minutes before we introduce the first tag
label_time = np.arange(4) if PDB else np.arange(3)
time_interval=5
label_time = time_interval*label_time
num_timepoints = len(label_time)
max_time_for_equilibrium = 50
save_folder = './'

class CustomTest(unittest.TestCase):
    def assertSeriesEqual(self, series1, series2, msg=None):
        try:
            assert_series_equal(series1, series2)
        except AssertionError as e:
            raise self.failureException(msg or str(e))

class Test_ElongEst(CustomTest):
    def __init__(self, methodName='Test_ElongEst', param=None):
        super().__init__(methodName)

    # def test_findA(self):
    #     x0 = np.array([0, 2, 5, 7,  10, 12, 15, 17,   18])
    #     x1 = np.array([2, 6, 8, 21, 11, 15, 18, 17.1, 30])
    #     endpoints = np.array([0, 5, 10, 15, 20, np.inf])
    #     A = np.array([[2, 0, 0, 0, 0],      # 0-2
    #                   [3, 1, 0, 0, 0],      # 2-6
    #                   [0, 3, 0, 0, 0],      # 5-8
    #                   [0, 3, 5, 5, 1],      # 7-21
    #                   [0, 0, 1, 0, 0],      # 10-11
    #                   [0, 0, 3, 0, 0],      # 12-15
    #                   [0, 0, 0, 3, 0],      # 15-18
    #                   [0, 0, 0, 0.1, 0],    # 17-17.1
    #                   [0, 0, 0, 2, 10]])    # 18-30
    #     pred_A = estElong.findA(x0, x1, endpoints)
    #     self.assertTrue(np.allclose(A, pred_A))
    #     x0 = np.array([0, 2, 5, 7,  10, 12, 15, 17,   18])
    #     x1 = np.array([2, 6, 8, 21, 11, 15, 18, 17.1, 30])
    #     endpoints = np.array([0, 5, 10, 15, 20, 20,np.inf])
    #     A = np.array([[2, 0, 0, 0,   0, 0],      # 0-2
    #                   [3, 1, 0, 0,   0, 0],      # 2-6
    #                   [0, 3, 0, 0,   0, 0],      # 5-8
    #                   [0, 3, 5, 5,   0, 1],      # 7-21
    #                   [0, 0, 1, 0,   0, 0],      # 10-11
    #                   [0, 0, 3, 0,   0, 0],      # 12-15
    #                   [0, 0, 0, 3,   0, 0],      # 15-18
    #                   [0, 0, 0, 0.1, 0, 0],    # 17-17.1
    #                   [0, 0, 0, 2,   0, 10]])    # 18-30
    #     pred_A = estElong.findA(x0, x1, endpoints)
    #     self.assertTrue(np.allclose(A, pred_A))

    # def test_filterA(self):
    #     A = np.array([[1, 2, 3], [0, 0, 0], [4, 5, 6], [0, 0, 0]])
    #     A = A.T
    #     pred_A = estElong.checkA_zero_columns(A)
    #     A_filtered = np.array([[1, 2, 3], [0, 0, 0], [4, 5, 6]])
    #     self.assertTrue(np.allclose(A_filtered.T, pred_A))
    #     return


    # def test_drop_trailing_zeros(self):
    #     df = pd.DataFrame({'a': [1, 2, 3, 0, 0, 0], 'b': [0, 0, 0, 1, 2, 0]})
    #     pred_df = drop_trailing_zeros(df)
    #     no_trail_zeros = pd.DataFrame({'a': [1, 2, 3, 0, 0], 'b': [0, 0, 0, 1, 2]})
    #     self.assertSeriesEqual(pred_df['a'], no_trail_zeros['a'])
    #     self.assertSeriesEqual(pred_df['b'], no_trail_zeros['b'])
    #     df = pd.DataFrame({'a': [1, 2, 3, 0, 0], 'b': [0, 0, 0, 1, 2]})
    #     pred_df = drop_trailing_zeros(df)
    #     no_trail_zeros = pd.DataFrame({'a': [1, 2, 3, 0, 0], 'b': [0, 0, 0, 1, 2]})
    #     self.assertSeriesEqual(pred_df['a'], no_trail_zeros['a'])
    #     self.assertSeriesEqual(pred_df['b'], no_trail_zeros['b'])

    #
    # def test_max_segments(self):
    #     gtf_df = sim.create_variable_gtf_df(nExons=2, length_fold_list=[1.65, 1.65, 1.65],
    #                                         elong_fold_list=[0.33, 0.33, 0.33], intronH_fold_list=[np.inf] * 3,
    #                                         SIM_FEAT_LEN=ONE_KB)
    #     coverage_df = pd.DataFrame({'position': [], 0:[]})
    #     elong_estimator = estElong(coverage_df, gtf_df)
    #     pred_max_segments = elong_estimator.find_max_segments()
    #     self.assertEqual(pred_max_segments, 5/0.2)

    # def test_regression_draw(self):
    #     gtf_df = sim.create_variable_gtf_df(nExons=2, length_fold_list=[1.65, 1.65, 1.65],
    #                                         elong_fold_list=[0.33, 0.33, 0.33], intronH_fold_list=[np.inf] * 3,
    #                                         SIM_FEAT_LEN=ONE_KB)
    #     elongf_df = pd.DataFrame({'start': np.arange(0, 4750, 200), 'txrate':  0.33})
    #     elongf_df['end'] = elongf_df['start'][1:].tolist() + [np.inf]
    #     gtf_df, elongf_df = sim.align_gtf_df_with_elongf_df(gtf_df, elongf_df)
    #     # exp_list = sim.generate_exp_given_one_gtf(gtf_df, elongf_df, save_folder=save_folder, label_time=label_time,
    #     #                                           target_exp=target_exp,
    #     #                                           num_total_transcript_millions=num_total_transcript_millions,
    #     #                                           lambda_init=lambda_init, burst_size=burst_size,
    #     #                                           wiggle_room=wiggle_room, eta_val=eta_val,
    #     #                                           insertsize_min=insertsize_min, insertsize_max=insertsize_max,
    #     #                                           read_length=read_length, simulate_cleavage=simulate_cleavage, PDB=PDB,
    #     #                                           max_time_for_equilibrium=max_time_for_equilibrium,
    #     #                                           frag_func=frag_func)
    #     # endpoint_df = viz.get_endpoints_across_time(exp_list)
    #     # coverage_df = viz.count_timeDep_read_coverage(exp_list[-1], endpoint_df, N=1, num_timepoints=len(exp_list))
    #     # coverage_df.to_csv('coverage_df.csv.gz', header = True, index=False, sep = '\t', compression='gzip')
    #     # gtf_df.to_csv('gtf_df.csv.gz', header=True, index=False, sep='\t', compression='gzip')
    #     coverage_df = pd.read_csv('coverage_df.csv.gz', header=0, index_col=None, sep='\t')
    #     coverage_df.columns = [0,1,2, 'position']
    #     gtf_df = pd.read_csv('gtf_df.csv.gz', header=0,index_col=None, sep='\t')
    #     elong_estimator = estElong(coverage_df, elongf_df = elongf_df, gtf_df= gtf_df, endT_idx=2, output_folder = save_folder)
    #     elong_estimator.draw_regression_lines(save_fn='test_estElong_regression.png', show=False)
    #     elong_estimator.draw_distance_travelled(save_fn='test_estElong_distance.png', show=False)
    #     self.assertTrue(os.path.exists('test_estElong_regression.png'))

    # def test_find_valid_positions(self):
    #     df = pd.DataFrame({'position': [1, 2, 3, 4, 5, 6], 0: [0, 0, 0, 1, 2, 0]})
    #     position = pd.Series([1, 2, 3, 4, 5])
    #     from estimates.elongation import find_valid_positions
    #     pred_position = find_valid_positions(df, 0)
    #     assert_series_equal(position, pred_position, check_names=False)
    #     return
    #
    def test_calculate_h(self):
        gtf_df = sim.create_variable_gtf_df(nExons=2, length_fold_list=[1.65, 1.65, 1.65],
                                            intronH_fold_list=[np.inf] * 3,
                                            SIM_FEAT_LEN=ONE_KB)
        elongf_df = pd.DataFrame({'start': np.arange(0, 4750, 200), 'txrate':  0.33})
        elongf_df['end'] = elongf_df['start'][1:].tolist() + [np.inf]
        gtf_df, elongf_df = sim.align_gtf_df_with_elongf_df(gtf_df, elongf_df)
        # exp_list = sim.generate_exp_given_one_gtf(gtf_df, elongf_df=elongf_df, save_folder=save_folder, label_time=label_time,
        #                                           target_exp=target_exp,
        #                                           num_total_transcript_millions=num_total_transcript_millions,
        #                                           lambda_init=lambda_init, burst_size=burst_size,
        #                                           wiggle_room=wiggle_room, eta_val=eta_val,
        #                                           insertsize_min=insertsize_min, insertsize_max=insertsize_max,
        #                                           read_length=read_length, simulate_cleavage=simulate_cleavage, PDB=PDB,
        #                                           max_time_for_equilibrium=max_time_for_equilibrium,
        #                                           frag_func=frag_func)
        # endpoint_df = viz.get_endpoints_across_time(exp_list)
        # coverage_df = viz.count_timeDep_read_coverage(exp_list[-1], endpoint_df, N=1, num_timepoints=len(exp_list))
        # coverage_df.to_csv('coverage_df.csv.gz', header = True, index=False, sep = '\t', compression='gzip')
        # gtf_df.to_csv('gtf_df.csv.gz', header=True, index=False, sep='\t', compression='gzip')
        coverage_df = pd.read_csv('coverage_df.csv.gz', header=0, index_col=None, sep='\t')
        coverage_df.columns = [0,1,2, 'position']
        gtf_df = pd.read_csv('gtf_df.csv.gz', header=0,index_col=None, sep='\t')
        elong_estimator = estElong(coverage_df, gtf_df, elongf_df=elongf_df, h_bin_bp= [200, 500, 1000], regress_bin_bp=200, output_folder = save_folder)
        elong_estimator.estimate()
        elong_estimator.save_estimates('pred_h.csv.gz')
        return


if __name__ == '__main__':
    unittest.main()