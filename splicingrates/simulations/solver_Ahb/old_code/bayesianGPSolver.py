import torch
import torch.nn as nn
import numpy as np
import gpytorch
from tqdm import tqdm
from gpytorch.models import ExactGP
from gpytorch.likelihoods import GaussianLikelihood
import time
universal_lr = 0.1
universal_num_epochs = 50
class BayesianLinearRegression(ExactGP):
    '''
    Let's say we have X\beta + epsilon = y, in which
    - epsilon ~ N(0, \sigma^2)
    - \beta ~ N(\mu, \sigma_0^2 * I)
    We want to find \beta that best fits the data and the prior information
    This can be interpreted as a Gaussian Process regression problem, with linear Kernel
    K(x,x') = x^T x' * \sigma_0^2
    Why? Let me document the math later
    '''
    def __init__(self, X, y, beta_mean, beta_var, likelihood=GaussianLikelihood()):
        '''
        :param X: (n, m) matrix
        :param y: (n,) vector
        :param beta_mean: tensor of size (m,) representing the mean of the prior distribution of beta
        :param beta_var: a scalar representing the variance of the prior distribution of beta
        '''
        super().__init__(X, y, likelihood)
        self.mean_module = gpytorch.means.LinearMean(input_size=X.shape[1])
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.LinearKernel()
        )
        # Set prior mean for beta
        self.mean_module.bias.data.fill_(0.0)  # Usually set to 0 as X usually includes a constant term
        self.mean_module.weights.data = beta_mean
        # Set prior variance for beta
        self.covar_module.base_kernel.variance = 1.0  # This should be fixed, because the beta_var is handled in ScaleKernel that wraps around this base kernel
        self.covar_module.outputscale = beta_var
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class CustomKernel(gpytorch.kernels.Kernel):
    def __init__(self, coordinates, X):
        super().__init__()
        self.coordinates = coordinates
        if len(self.coordinates.shape) == 1:
            self.coordinates = self.coordinates.unsqueeze(-1)  # (m, 1) to treat each coordinate as a 1D point
        self.X = X  # (n, m) where n: # samples, m: # features
        self.base_kernel = gpytorch.kernels.RBFKernel()  # the base_kernel.lengthscale is a learnable parameter bc its requires_grad is True
    def forward(self, x1, x2, diag=False, last_dim_is_batch=False, **params):
        # Compute RBF kernel for coordinates
        coord_covar = self.base_kernel(self.coordinates, self.coordinates)  # (m, m)
        # Compute X^T * RBF(coordinates) * X
        return self.X @ coord_covar @ self.X.t()

class BayesianLinearRegression_RBF(ExactGP):
    '''
    Let's say we have X\beta + epsilon = y, in which
    - epsilon ~ N(0, \sigma^2)
    - \beta ~ N(\mu, \Sigma)
    - \Sigma = RBFKernel(coordinates)  with parameters of the RBF kernel that is tunable
    We want to find \beta that best fits the data and the prior information
    This can be interpreted as a Gaussian Process regression problem, with linear Kernel
    K(x,x') = x^T \Sigma x
    Why? Let me document the math later
    '''
    def __init__(self, train_x, train_y, coordinates, prior_mean, likelihood=GaussianLikelihood()):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.LinearMean(input_size=train_x.size(-1))
        self.covar_module = CustomKernel(coordinates, train_x)
        self.mean_module.weights.data = prior_mean
        self.mean_module.bias.data.fill_(0.0)
    def forward(self, x):
        mean_y = self.mean_module(x)
        covar_y = self.covar_module(x, x)
        return gpytorch.distributions.MultivariateNormal(mean_y, covar_y)

class bayesian_solver:
    '''
    This model will take in data from A and b, solve for the A/h=b system of equations using the GP framework. The model will be trained using the BayesianLinearRegression model
    '''
    def __init__(self, A, b):
        '''
        Solve for the system of linear equations Ax = b
        :param A: (n, m) where n is the number of samples and m is the number of features (regions along the gene)
        :param b: (n,) where n is the number of samples --> the time it takes to traverse different segment of the gene
        '''
        self.train_x = A if type(A) == torch.Tensor else torch.tensor(A, dtype=torch.float32)
        self.train_y = b if type(b) == torch.Tensor else torch.tensor(b, dtype=torch.float32)

    def find_avg_hinv(self):
        '''
        Given A(n,m) and b(n,). Find h_inv (1,) such that A*h_inv = b by simply asking, on average, what h?
        We can do this by saying that rowsum(A)/h_avg = b --> find h_avg which should be just a scalar
        :return: h_inv
        '''
        rowsum_A = torch.sum(self.train_x, dim=1)
        h_inv = self.train_y/rowsum_A  # (n,1)
        return torch.mean(h_inv)  # (1,)

    @staticmethod
    def find_avg_x(A, b):
        '''
        Given A(n,m) and b(n,). Find x (1,) such that rowsum(A)x = b
        :param A:
        :param b:
        :return:
        '''
        rowsum_A = torch.sum(A, dim=1)
        x = b/rowsum_A  # (n,1)
        return torch.mean(x)  # (1,)

class bayesianLinear_solver(bayesian_solver):
    '''
    This model will solve A/h=b using the BayesianLinearRegression model
    '''
    def __init__(self, A, b, init_h=None):
        super().__init__(A, b)
        if init_h is None:
            init_h = self.estimate_h(A, b)
        init_h = torch.tensor(init_h, dtype=torch.float32) if type(init_h) != torch.Tensor else init_h
        if init_h.shape != torch.Size([A.shape[1]]):  # if init_h is a scalar, then make it a tensor (d,)
            init_h = torch.ones(A.shape[1]) * init_h
        self.init_h = init_h

    @staticmethod
    def estimate_h(A, b):
        '''
        Given A(n,m) and b(n,). Find mu (1,) such that A*mu = b
        Simply calculate the average of the values of h
        :param A:
        :param b:
        :return:
        '''
        A = torch.Tensor(A) if type(A) != torch.Tensor else A
        b = torch.Tensor(b) if type(b) != torch.Tensor else b
        rowsumA = torch.sum(A, dim=1)
        h = b/rowsumA  # (n,)
        h = torch.mean(h)  # (1,)
        return h # (1,) --> estimate h based on the average distance across the gene


    def solve(self, lr=universal_lr, num_epochs=universal_num_epochs):
        '''
        Solve for the system of linear equations Ax = b
        :param lr: learning rate
        :param num_epochs: number of epochs
        :param beta_variance: variance of the prior distribution of beta (h_inv). the covariance matrix, in this case, is beta_variance * I
        :return: x
        '''
        # Define the likelihood and model
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = BayesianLinearRegression(self.train_x.shape[1], init_h=self.init_h)
        model.train()
        likelihood.train()
        optimizer = (torch.optim.Adam(model.parameters(), lr=lr))  # params include mean and covariance parameters for the noise (likelihood) and the model ( parameters for the mean and covariance modules)
        # Define the loss (marginal log likelihood)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
        # Training loop
        for epoch in (range(num_epochs)):
            optimizer.zero_grad()
            output = model(self.train_x)
            loss = -mll(output, self.train_y)
            if epoch % 500 == 0:
                print('Epoch %d/%d - Loss: %.3f' % (epoch + 1, num_epochs, loss.item()))
            loss.backward()
            optimizer.step()
        # Switch to eval mode
        model.eval()
        likelihood.eval()
        pred_h = 1 / model.mean_module.weights.detach()  # (m,)
        return pred_h

class bayesianRBF_solver(bayesian_solver):
    '''
    This model will solve A/h=b using the BayesianLinearRegression_RBF model
    '''
    def __init__(self, A, b, h_bins=1):
        '''
        :param A:
        :param b:
        :param h_bins: bin size of each feature in the gene
        '''
        super().__init__(A, b)
        self.h_bins = h_bins

    def find_coordinates(self):
        '''
        Given A and h_bins, find coordinates for the RBF kernel
        :return: coordinates of the center point of each feature
        '''
        return torch.arange(0, self.train_x.shape[1]) * self.h_bins + self.h_bins / 2

    def solve(self, lr=universal_lr, num_epochs=universal_num_epochs):
        '''
        Solve for the system of linear equations Ax = b
        :param lr: learning rate
        :param num_epochs: number of epochs
        :return: x
        '''

        self.avg_hinv = self.find_avg_hinv()
        prior_mean = self.avg_hinv.clone().detach() * torch.ones(self.train_x.shape[1])
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        self.coordinates = self.find_coordinates()
        model = BayesianLinearRegression_RBF(self.train_x, self.train_y, coordinates=self.coordinates, prior_mean=prior_mean, likelihood=likelihood)
        model.train()
        likelihood.train()
        optimizer = (torch.optim.Adam(model.parameters(), lr=lr))  # params include mean and covariance parameters for the noise (likelihood) and the model ( parameters for the mean and covariance modules)
        # Define the loss (marginal log likelihood)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
        # Training loop
        for epoch in (range(num_epochs)):
            optimizer.zero_grad()
            output = model(self.train_x)
            loss = -mll(output, self.train_y)
            if epoch % 500 == 0:
                print('Epoch %d/%d - Loss: %.3f' % (epoch + 1, num_epochs, loss.item()))
            loss.backward()
            optimizer.step()
        # Switch to eval mode
        model.eval()
        likelihood.eval()
        pred_h = 1 / model.mean_module.weights.detach()  # (m,)
        return pred_h
