import torch
from torch import nn

class simpleModel(nn.Module):
    def __init__(self, d, init_h=1):
        super(simpleModel, self).__init__()
        self.d = d
        init_h = torch.tensor(init_h) if type(init_h) != torch.Tensor else init_h
        if init_h.shape != torch.Size([self.d]):
            init_h = torch.ones(d) * init_h
        self.h = nn.Parameter(init_h.clone().detach())

    def forward(self, X):
        h_inv = 1 / self.h
        return torch.matmul(X, h_inv)


