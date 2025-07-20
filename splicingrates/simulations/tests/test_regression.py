import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import unittest
import pandas as pd
from pandas.testing import assert_series_equal
from regression.piecewise_linear_regression import find_segment_index
from regression.piecewise_linear_regression import x_to_y_array
from regression.piecewise_linear_regression import y_to_x_array


class CustomTest(unittest.TestCase):
    def assertSeriesEqual(self, series1, series2, msg=None):
        try:
            assert_series_equal(series1, series2)
        except AssertionError as e:
            raise self.failureException(msg or str(e))

class TestPieceWiseLinearRegression(CustomTest):
    def __init__(self, methodName='TestPieceWiseLinearRegression', param=None):
        super().__init__(methodName)
        self.param = param

    def test_find_segment_index(self):
        px = np.array([0, 1, 2, 3])
        xs = np.array([  -0.5, 0.5, 1, 1.5, 2, 2.5, 3, 3.5])
        indices = np.array([0,   0, 1,   1, 2,   2, 2,   2])
        pred_idx = find_segment_index(xs, px)
        assert_series_equal(pd.Series(pred_idx), pd.Series(indices))
        px = np.array([3, 2, 1, 0])
        xs = np.array([  -0.5, 0.5, 1, 1.5, 2, 2.5, 3, 3.5])
        indices = np.array([2,   2, 1,   1, 0,   0, 0,   0])
        pred_idx = find_segment_index(xs, px)
        assert_series_equal(pd.Series(pred_idx), pd.Series(indices))
        px = np.array([0,1,1,2])
        xs = np.array([  -0.5, 0.5, 1, 1.5, 2, 2.5, 3, 3.5])
        indices = np.array([0,   0, 1,   2, 2,   2, 2,   2])
        pred_idx = find_segment_index(xs, px)
        assert_series_equal(pd.Series(pred_idx), pd.Series(indices))
        px = np.array([2, 1, 1, 0])
        xs = np.array([  -0.5, 0.5, 1, 1.5, 2, 2.5, 3, 3.5])
        indices = np.array([2,   2, 1,   0, 0,   0, 0,   0])
        pred_idx = find_segment_index(xs, px)
        assert_series_equal(pd.Series(pred_idx), pd.Series(indices))
        px = np.array([1, 1, 2, 3])
        xs = np.array([     0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5])
        indices = np.array([0,   0, 0,   1, 2,   2, 2,   2])
        pred_idx = find_segment_index(xs, px)
        assert_series_equal(pd.Series(pred_idx), pd.Series(indices))
        px = np.array([3, 2, 1, 1])
        xs = np.array([     0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5])
        indices = np.array([2,   2, 2,   1, 0,   0, 0,   0])
        pred_idx = find_segment_index(xs, px)
        assert_series_equal(pd.Series(pred_idx), pd.Series(indices))

    def test_x_to_y(self):
        px = np.array([0, 1, 2, 3])
        py = np.array([2, 1, 1, 0])
        xs =          np.array([-0.5, 0.5, 1, 1.5, 2, 2.5, 3, 3.5])
        ys =          np.array([2.5, 1.5,             0.5, 0,-0.5])
        xs_filtered = np.array([-0.5, 0.5,            2.5, 3, 3.5])
        # the x_to_y_array function filters out the values of x and y that corresponds to points lying on a horizontal line on the xy plane
        # therefore, it will get rid of points on [1,2] on the x axis and [1,1] on the y axis
        pred_ys, pred_xs_filtered = x_to_y_array(xs, px, py)
        assert_series_equal(pd.Series(pred_ys), pd.Series(ys))
        assert_series_equal(pd.Series(pred_xs_filtered), pd.Series(xs_filtered))

    def test_y_to_x(self):
        px = np.array([0, 1, 2, 3])
        py = np.array([2, 1, 1, 0])
        ys = np.array([2.5,  2, 1.5,      1, 0.5, 0, -0.5])
        xs = np.array([-0.5, 0, 0.5, np.nan, 2.5, 3,  3.5])
        pred_xs = y_to_x_array(ys, px, py)
        assert_series_equal(pd.Series(pred_xs), pd.Series(xs))


if __name__ == '__main__':
    unittest.main()