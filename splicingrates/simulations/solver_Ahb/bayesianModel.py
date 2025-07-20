import torch
from torch import nn
from torch.nn import functional as F
import gpytorch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class bayesianNormal(nn.Module):
    def __init__(self, d, init_h = 1.0, p_sigma_h = 0.5):
        super(bayesianNormal, self).__init__()
        self.d = d
        init_h = torch.tensor(init_h, dtype=torch.float32) if type(init_h) != torch.Tensor else init_h
        if init_h.shape != torch.Size([d]):
            init_h = torch.ones(d) * init_h
        self.init_h = init_h  # (d,)
        # variational distribution of h
        self.q_mu_h = nn.Parameter(init_h.clone().detach(), requires_grad=True)
        self.q_sigma_h = nn.Parameter(torch.tensor(p_sigma_h), requires_grad=True)
        # prior distribution of h
        self.p_mu_h = init_h.clone().detach().to(device=device) # prior mean of h. (d,)
        self.p_sigma_h = torch.tensor(p_sigma_h, dtype=torch.float32, device=device) # prior variance of h. (d,) or a scalar

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
        epsilon = torch.randn(mu.shape, device=device)
        return mu + F.softplus(sigma) * epsilon

    def forward(self, X):
        '''
        :param X: torch tensor of shape (n,d)
        :return: predicted y  (n,) given the current values of q_mu_h and q_sigma_h
        '''
        q_h = self.reparameterize_normal(self.q_mu_h, self.q_sigma_h)
        y_pred = torch.matmul(X, 1/q_h)
        return y_pred

class bayesian_RBF(nn.Module):
    def __init__(self, d, init_h = 1.0, coords = None, rbf_sigma= 1):
        super(bayesian_RBF, self).__init__()
        self.d = d
        init_h = torch.tensor(init_h, dtype=torch.float32) if type(init_h) != torch.Tensor else init_h
        if init_h.shape != torch.Size([d]):
            init_h = torch.ones(d) * init_h
        self.init_h = init_h.to(device=device)  # (d,)
        # get the covariance matrix for the h's based on RBF kernel of the coordinates
        if coords is None:
            coords = torch.arange(d).view(-1, 1)  # (d,1)
        else:
            coords = torch.tensor(coords, dtype=torch.float32).view(-1, 1)  # (d, 1)
        coords = coords.to(device=device)
        self.Sigma_RBF = self.calculate_Sigma_RBF(coords, rbf_sigma)  # (d,d) A Gpytorch object
        # prior distribution of h
        self.p_mu_h = init_h.clone().detach().to(device=device)  # prior mean of h. (d,)
        # variational distribution of h
        self.q_mu_h = nn.Parameter(init_h.clone().detach(), requires_grad=True)

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
        covar_module = gpytorch.kernels.RBFKernel().to(device=device)
        covar_module.lengthscale = torch.tensor(rbf_sigma).to(device=device)  # Different lengthscales for each input dimension
        covar_module.raw_lengthscale.requires_grad = False
        Sigma_RBF = covar_module(coords)
        return Sigma_RBF

    def forward(self, X):
        '''
        :param X: torch tensor of shape (n,d)
        :return: predicted y  (n,) given the current values of q_mu_h and q_sigma_h
        '''
        mvn = gpytorch.distributions.MultivariateNormal(self.q_mu_h, self.Sigma_RBF)
        q_h = mvn.rsample()  # this will take care of reparameterization for me
        y_pred = torch.matmul(X, 1/q_h)
        return y_pred


class bayesian_logNormal(nn.Module):
    def __init__(self, d, init_h=1, p_sigma_h = 0.5):
        '''
        Solve for the system of linear equations A/h = b
        We have the following assumption:
        - A/h + \epsilon = b, in which \epsilon ~ N(0, \sigma^2)
        - prior h ~ logNormal(\mu_h, \sigma_h)
        - varational distribution of h ~ logNormal(q_mu_h, q_sigma_h)
        :param A: (n, m) where n is the number of samples and m is the number of features (regions along the gene)
        :param b: (n,) where n is the number of samples --> the time it takes to traverse different segment of the gene
        :param init_h: initial value of h.
        '''
        super(bayesian_logNormal, self).__init__()
        self.d = d
        init_h = torch.tensor(init_h, dtype=torch.float32) if type(init_h) != torch.Tensor else init_h
        if init_h.shape != torch.Size([d]):
            init_h = torch.ones(d) * init_h
        self.init_h = init_h.to(device=device)  # (d,)
        mu = self.mu_given_h(init_h, p_sigma_h)
        sigma = torch.tensor(p_sigma_h, dtype=torch.float32)
        # prior distribution of h
        self.p_mu_h = mu.clone().detach().to(device=device)  # prior mean of h. (d,)
        self.p_mu_h.requires_grad = False
        self.p_sigma_h = sigma.clone().detach().to(device=device) # prior variance of h. (d,) or a scalar
        self.p_sigma_h.requires_grad = False
        # parameters of the model
        self.q_mu_h = nn.Parameter(mu.clone().detach(), requires_grad=True) # variational mean of h. (d,)
        self.q_sigma_h = nn.Parameter(sigma.clone().detach(), requires_grad=True) # variational variance of h. (d,) or a scalar

    @staticmethod
    def mu_given_h(h, sigma_h):
        '''
        :param h: (d,) tensor of the predicted h values
        :param sigma_h: (d,) tensor or a scalar of the predicted sigma values
        :return: mu: (n,) tensor of the predicted y values. The model assumes that h ~ logNormal(\mu_h, \sigma_h)
        if h ~ logNormal(mu_h, sigma_h), then E(h) = exp(mu_h + 0.5 * sigma_h^2)
        '''
        if type(sigma_h) == torch.Tensor:
            pass
        else:
            sigma_h = torch.tensor(sigma_h, dtype=torch.float32)
        if sigma_h.shape == torch.Size([]):
            sigma_h = torch.ones(h.shape) * sigma_h
        elif sigma_h.shape == torch.Size([1]):
            sigma_h = torch.ones(h.shape) * sigma_h
        else:
            pass
        mu = torch.log(h) - 0.5 * sigma_h**2
        return mu

    @staticmethod
    def reparametrize_log_normal(mu, sigma):
        '''
        Reparametrize the log-normal distribution.
        :param mu: Mean of the log-normal distribution.
        :param sigma: Standard deviation of the log-normal distribution. Can be just a scalar or the same shape as mu.
        :return: A sample from the log-normal distribution.
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
        epsilon = torch.randn(mu.shape, device=mu.device)
        return torch.exp(mu + F.softplus(sigma) * epsilon)
        # softplus(sigma) = log(1 + exp(sigma))

    def forward(self, X):
        '''

        :return: predicted y and the values of the q_mu_beta
        '''
        # given the current q_mu_h and q_sigma_h, reparametrize the logNormal distribution to generate sample h
        # q_h = exp(q_mu_h + softplus(q_sigma_h) * epsilon)  where epsilon ~ N(0, 1)
        q_h = self.reparametrize_log_normal(self.q_mu_h, self.q_sigma_h) # we sample h from the variational distribution
        # calculate the predicted y
        y_pred = torch.matmul(X, 1/q_h)
        return y_pred
