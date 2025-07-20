import torch
import torch.nn as nn
import numpy as np


class TimeCalculator(nn.Module):
    def __init__(self, A, b, init_lasth=torch.tensor([])):
        '''
        Solve for the system of linear equations Ax = b
        :param A:
        :param b:
        '''
        super(TimeCalculator, self).__init__()
        if isinstance(A, np.ndarray):
            A = torch.tensor(A, dtype=torch.float32)
        if isinstance(b, np.ndarray):
            b = torch.tensor(b, dtype=torch.float32)
        if isinstance(init_lasth, np.ndarray):
            init_lasth = torch.tensor(init_lasth, dtype=torch.float32)
        self.A = A.clone().detach().float()
        self.b = b.clone().detach().float()
        # if b is of size (n,) then make it (n,1)
        if len(self.b.shape) == 1:
            self.b = self.b.unsqueeze(1)
        assert self.A.requires_grad == False
        assert self.b.requires_grad == False
        self.n = A.shape[0]
        self.m = A.shape[1]
        self.m1 = self.m - init_lasth.shape[0]  # first m1 variables should be tuned
        self.m2 = self.m - self.m1  # last m2 variables should be fixed
        self.A_for_rand = self.A[:, self.m1]  # subset of A that is for multiplying with the parameters that we randomly initialize
        self.A_for_init = self.A[:, self.m1:]  # subset of A that is for multiplying with the parameters that we initialize with the last h
        self.init_lasth = init_lasth.clone().detach().float()
        # Initialize h_inv as a parameter with size (m, 1)
        self.h_inv = nn.Parameter(torch.empty(self.m, 1, dtype=torch.float32))
        self.init_model_params()

    def init_model_params(self):
        '''
        Initialize the model parameters h_inv such that the first m1 variables are initialized randomly and the last m2 variables are initialized with the last h
        :return:
        '''
        # Initialize the first m1 entries randomly
        nn.init.normal_(self.h_inv[:self.m1], mean=0, std=0.1)
        # Initialize the last m2 entries with the provided init_lasth values
        if len(self.init_lasth) > 0:
            self.h_inv.data[self.m1:] = self.init_lasth.view(-1, 1)
        return

    def forward(self, x):
        return torch.mm(self.A, self.h_inv)


# Example usage:
A = np.random.rand(10, 8)
b = np.random.rand(10)
init_lasth = np.array([0.5, 0.6, 0.7])

model = TimeCalculator(A, b, init_lasth)

# Example input for the forward pass
x = torch.randn(8, 1, dtype=torch.float32)

# Forward pass
output = model(x)
print("Model output:", output)

# Define a loss function
criterion = nn.MSELoss()

# Calculate loss
loss = criterion(output, model.b)
print("Loss:", loss.item())
