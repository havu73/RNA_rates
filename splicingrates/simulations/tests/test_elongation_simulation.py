# this file will tests some the functions that caluclate the different features of SIMULATION for the elongation rates
# such as calculating the time it takes for transcripts to traverse from point A to poin B along the gene,
# calculating the end point at m minutes later of the labeling experiment
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
from numpy.testing import assert_almost_equal
import unittest
import pandas as pd
from transcription.elongation_calculation import find_feature_overlap, get_endpoints_from_gtf, time_to_elongate
import transcription.helper as helper
from pandas.testing import assert_series_equal

class CustomTest(unittest.TestCase):
    def assertSeriesEqual(self, series1, series2, msg=None):
        try:
            assert_series_equal(series1, series2)
        except AssertionError as e:
            raise self.failureException(msg or str(e))

class TestCheckElongationCalculation(CustomTest):
    def __init__(self, methodName='check_elongation_calculation', param=None):
        super().__init__(methodName)
        self.elongf_df = pd.DataFrame({'txrate': [1, 2, 1], 'start': np.array([0, 5, 10]) * helper.ONE_KB, 'end': np.array([5, 10, 15]) * helper.ONE_KB})
        self.elongf_df['feature'] = 'elong_feat'
        self.elongf_df.loc[len(self.elongf_df)] = {'txrate': 1, 'start':15000, 'end': 15001, 'feature': 'PAS'}
        self.elongf_df.loc[len(self.elongf_df)] = {'txrate': 1, 'start':15001, 'end': np.inf, 'feature': 'RTR'}
        # print (self.elongf_df)

    def test_find_feature_overlap(self):
        feat_enpoints = get_endpoints_from_gtf(self.elongf_df, convert_to_KB=False)
        num_feat = len(feat_enpoints)-1
        right_endpoints = np.array([0,5000,10000,15000,15001, np.inf])
        assert np.array_equal(feat_enpoints, right_endpoints)
        assert np.array_equal(find_feature_overlap(0,0, feat_enpoints), np.array([0] * (num_feat)))
        assert np.array_equal(find_feature_overlap(0,2500, feat_enpoints), np.array([2500] + [0] * (num_feat-1)))
        assert np.array_equal(find_feature_overlap(4999, 25002, feat_enpoints), np.array([1,5000,5000,1,10001]))
        assert np.array_equal(find_feature_overlap(20000, 20001,feat_enpoints), np.array([0,0,0,0,1]))

    def test_time_to_elongate(self):
        assert_almost_equal(time_to_elongate(9000, 13000, self.elongf_df, 'txrate'),  3.5)
        assert_almost_equal(time_to_elongate(0, 17000, self.elongf_df, 'txrate'), 14.5)
        assert_almost_equal(time_to_elongate(10000, 17999, self.elongf_df, 'txrate'), 7.999)
        assert_almost_equal(time_to_elongate(3500, 6000, self.elongf_df, 'txrate'), 2)

    def test_calculate_enlongated_endsite(self):
        from transcription.elongation_calculation import calculate_enlongated_endsite
        assert_almost_equal(calculate_enlongated_endsite(0, self.elongf_df, 5), 5000)  # prev_stop, elong_df, time_since_prev
        assert_almost_equal(calculate_enlongated_endsite(2500, self.elongf_df, 5), 10000)
        assert_almost_equal(calculate_enlongated_endsite(10000, self.elongf_df, 5), 15000)
        assert_almost_equal(calculate_enlongated_endsite(15000, self.elongf_df, 5), 20000)

    def test_calculate_startsite_given_pred_h(self):
        from transcription.elongation_calculation import calculate_startsite_given_pred_h
        assert_almost_equal(calculate_startsite_given_pred_h(5000, self.elongf_df, e_colname='txrate', time_since_prev = 5), 0)
        assert_almost_equal(calculate_startsite_given_pred_h(10000, self.elongf_df, e_colname='txrate', time_since_prev = 5), 2500)
        assert_almost_equal(calculate_startsite_given_pred_h(17000, self.elongf_df, e_colname='txrate', time_since_prev=5), 12000)
    # def test_splice_prob(self):
    #     from transcription.transcripts import calculate_splice_prob
    #     row = self.elongf_df.iloc[1]  # the intron
    #     assert_almost_equal( calculate_splice_prob(endpoint= 10000, row=row, elongf_df=self.elongf_df), 0)
    #     assert calculate_splice_prob(endpoint= 20000, row=row, elongf_df=self.elongf_df) == 0.5


if __name__ == '__main__':
    unittest.main()