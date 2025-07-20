# this code is supposed to help debug some of the issues with the system to do piece-wise linear regression and calculation og elongation rates given the data of the read coverage
# the code is not tested, it is very ad-hoc and will need refinement but it is not a priority at the moment
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import unittest
import pandas as pd
from pandas.testing import assert_series_equal
from regression.fixedX_pwlr import find_increase_ranges

class CustomTest(unittest.TestCase):
    def assertSeriesEqual(self, series1, series2, msg=None):
        try:
            assert_series_equal(series1, series2)
        except AssertionError as e:
            raise self.failureException(msg or str(e))

class Test_ElongEst(CustomTest):
    def __init__(self, methodName='test_fixedX_pwlr', param=None):
        super().__init__(methodName)

    def test_find_increase_ranges(self):
        py = torch.tensor([4,3,2,1])
        exp_ranges = []
        print('test 1: ', find_increase_ranges(py))
        self.assertEqual(find_increase_ranges(py), exp_ranges)
        py = torch.tensor([1,2,3,4])
        exp_ranges = [(1, 4)]
        print('test 2: ', find_increase_ranges(py))
        self.assertEqual(find_increase_ranges(py), exp_ranges)
        py = torch.tensor([0, -1, -2,-1,-1.5,-2, -3,0, -4])
        exp_ranges = [(-2,-1), (-3,0)]
        print('test 3: ', find_increase_ranges(py))
        self.assertEqual(find_increase_ranges(py), exp_ranges)


if __name__ == '__main__':
    unittest.main()