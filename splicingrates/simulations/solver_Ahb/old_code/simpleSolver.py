import torch
from torch import nn
import numpy as np
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR
class TimeCalculator(nn.Module):
    def __init__(self, A, b, init_lasth=torch.tensor([])):
        '''
        Solve for the system of linear equations Ax = b
        :param A:
        :param b:
        '''
        super(TimeCalculator, self).__init__()
        if type(A) == np.ndarray:
            A = torch.tensor(A, dtype=torch.float32)
        if type(b) == np.ndarray:
            b = torch.tensor(b, dtype=torch.float32)
        if type(init_lasth) == np.ndarray:
            init_lasth = torch.tensor(init_lasth, dtype=torch.float32)
        self.A = A.clone().detach()
        self.b = b.clone().detach()
        # if b is if size (n,) then make it (n,1)
        if len(self.b.shape) == 1:
            self.b = self.b.unsqueeze(1)
        assert self.A.requires_grad == False
        assert self.b.requires_grad == False
        self.n = A.shape[0]
        self.m = A.shape[1]
        self.m1 = self.m-init_lasth.shape[0] # first m1 variables should be tuned
        self.m2 = self.m-self.m1  # last m2 variables should be fixed
        self.A_for_rand = self.A[:, :self.m1]  # subset of A that is for multiplying with the parameters that we randomly initialize
        self.A_for_init = self.A[:, self.m1:]  # subset of A that is for multiplying with the parameters that we initialize with the last h
        self.init_lasth = init_lasth.clone().detach().float()
        self.init_model_params()

    def init_model_params(self):
        '''
        Initialize the model parameters h_inv such that the first m1 variables are initialized randomly and the last m2 variables are initialized with the last h
        :return:
        '''
        if self.init_lasth.ndim == 1:
            self.init_lasth = self.init_lasth.unsqueeze(1)
        self.h_inv = nn.Parameter(torch.empty(self.m,1, dtype=torch.float32))
        nn.init.normal_(self.h_inv, mean=0, std=0.1)
        if len(self.init_lasth) > 0:
            self.h_inv.data = torch.cat((self.h_inv[:self.m1], 1/self.init_lasth), 0)
        return

    def forward(self):
        '''
        Compute the loss function
        :param h_inv_grad:
        :return: calculate the time it takes to go through each segment in A
        '''
        time = torch.mm(self.A, self.h_inv)
        return time

class simpleSolver:
    def __init__(self, A, b, init_lasth=torch.tensor([])):
        '''
        Solve for the system of linear equations Ax = b
        :param A:
        :param b:
        '''
        model = TimeCalculator(A, b, init_lasth)
        self.model = model
        self.target = b if type(b) == torch.Tensor else torch.tensor(b, dtype=torch.float32)
        if len(self.target.shape) == 1:
            self.target = self.target.unsqueeze(1)


    def solve(self, lr=0.01, num_epochs=1000):
        '''
        Solve for the system of linear equations Ax = b
        :return: x
        '''
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        scheduler = CosineAnnealingLR(optimizer, T_max=100)
        loss_fn = torch.nn.MSELoss()
        for epoch in (range(num_epochs)):
            optimizer.zero_grad()  # Zero the gradients
            time = self.model()
            loss = loss_fn(time, self.target)
            loss.backward()  # Backpropagate the loss
            optimizer.step()  # Update h
            if epoch % 500 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item()}")
            scheduler.step()  # update the learning rate based on the scheduler
        # append the lasth to h
        return (1/self.model.h_inv).detach()

class autoR_Ahb:
    '''
    This is a class that I created with the hope that I can solve the system A/h=b in a naive way that I tend to solve using the back substitution that I learned to do in K-12 school.
    I have tested it one a few cases and I realized that it does not work well for my case. So, don't use it
    '''
    def __init__(self, A, b):
        '''
        Solve for the system of linear equations Ax = b, specifically fit for RNA elongation rate estimation
        :param A:
        :param b:
        '''
        if type(A) == np.ndarray:
            A = torch.tensor(A, dtype=torch.float32)
        if type(b) == np.ndarray:
            b = torch.tensor(b, dtype=torch.float32)
        self.A = A
        self.b = b
        self.n = A.shape[0]
        self.m = A.shape[1]
        self.curr_lasth = torch.tensor([])

    @staticmethod
    def subsetA(A, b, leadZ_m = 0):
        '''
        return subset of A where the first m columns are all 0, we do not care about the values in columns that follow the first m columns
        assumption A are all >=0
        :param A:
        :param m:
        :return:
        '''
        # screen each row of A to see if the first m columns are all 0
        mask = torch.sum(A[:, :leadZ_m], dim=1) == 0
        result_A = A[mask, :]
        result_b = b[mask]
        # get rid of the first leadZ_m columns of result
        return result_A[:, leadZ_m:], result_b

    def _solve_subset(self, last_m=0, lasth=torch.tensor([])):
        '''
        Solve for the system of linear equations Ax = b
        :param A:
        :param b:
        :return: x
        '''
        subsetA, subsetb = autoR_Ahb.subsetA(self.A, self.b, leadZ_m=self.m-last_m)
        solver = simpleSolver(subsetA, subsetb, lasth)
        return solver.solve(num_epochs=3000)

    def solve(self, last_m=0, lasth=torch.tensor([]), lr=0.01, num_epochs=1000):
        '''
        Solve for the system of linear equations Ax = b
        :return: x
        '''
        self.curr_lasth = torch.tensor([], dtype=torch.float32)
        for last_m in range(len(self.curr_lasth)+1, self.m+1):
            result = self._solve_subset(last_m=last_m, lasth=self.curr_lasth)
            # add the first value in result to lasth
            self.curr_lasth = result.detach().clone()
            # self.curr_lasth = torch.cat((torch.tensor([result[0]]), self.curr_lasth), 0)
            print(f"last_m: {last_m}, lasth: {self.curr_lasth}")
            print(f"result: {self.curr_lasth}")
        import pdb; pdb.set_trace()
        return self.curr_lasth

