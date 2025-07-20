import torch
from torch import nn
from torch.nn import functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.distributions as dist
import gpytorch
universal_lr = 0.1
universal_num_epochs = 50

class bayesianNormal(nn.Module):
    def __init__(self, d, init_h = 1.0, p_sigma_h = 0.5):
        super(bayesianNormal, self).__init__()
        init_h = torch.tensor(init_h, dtype=torch.float32) if type(init_h) != torch.Tensor else init_h
        if init_h.shape != torch.Size([self.train_x.shape[1]]):
            init_h = torch.ones(d) * init_h
        self.init_h = init_h  # (d,)
        # prior distribution of h
        self.p_mu_h = init_h.clone.detach() # prior mean of h. (d,)
        self.p_sigma_h = torch.tensor(p_sigma_h, dtype=torch.float32) # prior variance of h. (d,) or a scalar
        # variational distribution of h
        self.q_mu_h = nn.Parameter(init_h.clone().detach())
        self.q_sigma_h = nn.Parameter(torch.tensor(p_sigma_h, dtype=torch.float32, requires_grad=True))

    @staticmethod
    def reparameterize_normal(mu, sigma):
        '''
        Reparametrize the normal distribution.
        :param mu: Mean of the normal distribution.
        :param sigma: Standard deviation of the normal distribution. Can be just a scalar or the same shape as mu.
        :return: A sample from the normal distribution.
        '''
        # check if sigma is a tensor of the same shape as mu or a scalar tensor
        if sigma.shape == mu.shape:
            pass
        elif sigma.shape == torch.Size([]):
            pass
        elif sigma.shape == torch.Size([1]):
            pass
        else:
            raise ValueError('sigma must be either a scalar or a tensor of the same shape as mu')
        epsilon = torch.randn(mu.shape)
        return mu + F.softplus(sigma) * epsilon

    def forward(self, X):
        '''
        :param X: torch tensor of shape (n,d)
        :return: predicted y  (n,) given the current values of q_mu_h and q_sigma_h
        '''
        q_h = self.reparameterize_normal(self.q_mu_h, self.q_sigma_h)
        y_pred = torch.matmul(X, 1/q_h)
        return y_pred

class Solver():
    def __init__(self, dataloader, n, d, init_h, p_sigma_h=0.5):
        self.dataloader = dataloader

    def loss(self, y_pred, y, *args, **kwargs):
        pass

    def solve(self, lr=universal_lr, num_epochs=universal_num_epochs):
        pass

class bayesianNormal_solver():
    def __init__(self, A, b, init_h, sigma_eps=0.1):
        self.train_x = A
        self.train_y = b
        d = self.train_x.shape[1]
        n = self.train_x.shape[0]
        self.model = bayesianNormal(d, init_h)
        self.epsilon = dist.MultivariateNormal(torch.zeros(n), torch.eye(n) * sigma_eps ** 2)  # noise distribution

    def elbo_loss(self, y_pred, q_mu_h, q_sigma_h):
        '''
        Calculate the ELBO loss function. For this model:
        - likelihood: p(y|A,h) = N(y|A/h, \sigma^2)
        - prior: p(h) = logNormal(h|mu_h, sigma_h)
        - variational: q(h) = logNormal(h|q_mu_h, q_sigma_h)
        - ELBO = E_q[log p(y|A,h)] - KL[q(h)||p(h)] where
        - E_q[log p(y|A,h)] = \sum_i log N(y_i|A_i/h, \sigma^2), where A_i/h is the y_pred and \sigma^2 is the variance of the noise
        - KL[q(h)||p(h)] = 0.5 * \sum_i (1 + log q_sigma_h[i]^2 - q_mu_h[i]^2 - q_sigma_h[i]^2 - log p_sigma_h[i]^2)
        :param y_pred:
        :param q_mu_h:
        :param q_sigma_h:
        :return:
        '''
        # calculate the negative log likelihood of y_pred and y, given that the difference should follow epsilon ~ N(0, \sigma^2)
        nll = -self.epsilon.log_prob(y_pred - self.train_y).mean()
        # calculate the KL divergence between q(h)~lognormal(q_mu_h, q_sigma_h) and p(h)~lognormal(p_mu_h, p_sigma_h)
        d = q_mu_h.shape[0]
        kl_divergence = d*0.5 * (torch.log(self.p_sigma_h**2)-torch.log(q_sigma_h**2) + (q_sigma_h**2/self.p_sigma_h**2) - 1) + \
              0.5 * (1/q_sigma_h**2) * torch.sum((q_mu_h-self.p_mu_h)**2)
        return nll + kl_divergence


    def solve(self, lr=universal_lr, num_epochs=universal_num_epochs):
        '''
        Solve for the system of linear equations Ax = b
        :param lr: learning rate
        :param num_epochs: number of epochs
        :param beta_variance: variance of the prior distribution of beta (h_inv). the covariance matrix, in this case, is beta_variance * I
        :return: x
        '''
        # Define the likelihood and model
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




class bayesian_RBF(nn.Module):
    def __init__(self, d, init_h = 1.0, coords = None, rbf_sigma= 1):
        super(bayesian_RBF, self).__init__()
        init_h = torch.tensor(init_h, dtype=torch.float32) if type(init_h) != torch.Tensor else init_h
        if init_h.shape != torch.Size([self.train_x.shape[1]]):
            init_h = torch.ones(d) * init_h
        self.init_h = init_h  # (d,)
        # get the covariance matrix for the h's based on RBF kernel of the coordinates
        if coords is None:
            coords = torch.arange(d).view(-1, 1)  # (d,1)
        else:
            coords = torch.tensor(coords, dtype=torch.float32).view(-1, 1)  # (d, 1)
        self.Sigma_RBF = self.calculate_Sigma_RBF(coords, rbf_sigma)  # (d,d)
        # prior distribution of h
        self.p_mu_h = init_h.clone.detach() # prior mean of h. (d,)
        # variational distribution of h
        self.q_mu_h = nn.Parameter(init_h.clone().detach())

    @staticmethod
    def calculate_Sigma_RBF(coords, rbf_sigma=1):
        '''
        Given the coordinate along the gene of each of the windows for which we are trying to calculate elongation rates, calculate the kernel matrix between each position.
        This martrix represents the covariance between each of the elongation rates.
        :param coords: tensor of shape (d,1) --> the coordinates of the windows for which we are trying to calculate elongation rates
        :param rbf_sigma: sigma values for the RBF kernel
        :return:
        '''
        if type(coords) != torch.Tensor:
            raise ValueError('The coordinates of the windows for which we are trying to calculate elongation rates must be provided')
        d = coords.shape[0]
        Sigma_RBF = torch.eye(d)
        for i in range(d):
            for j in range(i+1,d):
                rbf = torch.exp(-torch.norm(coords[i] - coords[j])**2 / (2 * rbf_sigma**2))
                Sigma_RBF[i, j] = rbf
                Sigma_RBF[j, i] = rbf
        return Sigma_RBF

    def forward(self, X):
        '''
        :param X: torch tensor of shape (n,d)
        :return: predicted y  (n,) given the current values of q_mu_h and q_sigma_h
        '''
        mvn = gpytorch.distributions.MultivariateNormal(self.q_mu_h, self.Sigma_lazy)
        q_h = mvn.rsample()  # this will take care of reparameterization for me
        y_pred = torch.matmul(X, 1/q_h)
        return y_pred

