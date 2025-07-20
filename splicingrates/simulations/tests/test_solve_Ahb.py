# this code is supposed to help debug some of the issues with the system to do piece-wise linear regression and calculation og elongation rates given the data of the read coverage
# the code is not tested, it is very ad-hoc and will need refinement but it is not a priority at the moment
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import unittest
import pandas as pd
from pandas.testing import assert_series_equal
from solver_Ahb.autoR_solver import simpleSolver, autoR_Ahb
from estimates.elongation import estElong

class CustomTest(unittest.TestCase):
    def assertSeriesEqual(self, series1, series2, msg=None):
        try:
            assert_series_equal(series1, series2)
        except AssertionError as e:
            raise self.failureException(msg or str(e))

class Test_AutoR(CustomTest):
    def __init__(self, methodName='test_solver_Ahb', param=None):
        super().__init__(methodName)
        fn = 'x0_x1.csv.gz'
        df = pd.read_csv(fn, header=0, index_col=None, sep='\t', compression='gzip')
        self.x0 = torch.tensor(df['x0'].values)
        self.x1 = torch.tensor(df['x1'].values)
        self.endpoints = torch.arange(0, 4.95, 0.2)
        inf_tensor = torch.tensor([float('inf')])
        self.endpoints = torch.cat((self.endpoints, inf_tensor), 0)
        self.A = estElong.findA(self.x0, self.x1, self.endpoints) # it is a tensor of shape (n, m)
        # b is a tensor of the same legnth as A, with each entry being 5
        self.b = torch.tensor([5] * len(self.A), dtype=torch.float32)

    def test_simpleSolver(self):
        solver = simpleSolver(self.A, self.b)
        h1 = solver.solve( num_epochs=10000)
        # h2 = estElong.solve_Ahinv_b(self.A, self.b)
        # print('h2: ', h2)
        print('h1: ', h1)
        # assert len(h1) == len(h2)

    # def test_simpleSolver_with_lasth(self):
    #     lasth = torch.tensor([[0.33]])
    #     solver = simpleSolver(self.A, self.b, lasth)
    #     h_inv = solver.solve()
    #     print('h_inv: ', h_inv)

    # def test_solve_Ahb(self):
    #     solver = autoR_Ahb(self.A, self.b)
    #     h_inv = solver.solve()
    #     print('h_inv: ', h_inv)



