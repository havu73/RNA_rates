import torch
from torch import nn
from torch.nn import functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
class bayesian_VI_Ahb(nn.Module):
    def __init__(self, A, b, init_h, p_sigma_h = 0.5):
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
        super(bayesian_VI_Ahb, self).__init__()
        self.train_x = A if type(A) == torch.Tensor else torch.tensor(A, dtype=torch.float32)  # (n,d)
        self.train_y = b if type(b) == torch.Tensor else torch.tensor(b, dtype=torch.float32)  # (n,)
        init_h = torch.tensor(init_h, dtype=torch.float32) if type(init_h) != torch.Tensor else init_h
        if init_h.shape != torch.Size([self.train_x.shape[1]]):
            init_h = torch.ones(self.train_x.shape[1]) * init_h
        self.init_h = init_h  # (d,)
        mu = self.mu_given_h(init_h, p_sigma_h)
        # prior distribution of h
        self.p_mu_h = torch.tensor(mu, dtype=torch.float32)  # prior mean of h. (d,)
        self.p_sigma_h = torch.tensor(p_sigma_h, dtype=torch.float32) # prior variance of h. (d,) or a scalar
        # parameters of the model
        self.q_mu_h = nn.Parameter(torch.tensor(mu, dtype=torch.float32, requires_grad=True)) # variational mean of h. (d,)
        self.q_sigma_h = nn.Parameter(torch.tensor(p_sigma_h, dtype=torch.float32, requires_grad=True)) # variational variance of h. (d,) or a scalar

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
        epsilon = torch.randn(mu.shape)
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
        return y_pred, self.q_mu_h, self.q_sigma_h  # (n,), (d,), (d,) --> needed to calcualte the loss function

class bayesian_logNormal_solver():
    def __init__(self, A, b, init_h=None, p_sigma_h = 0.5, sigma_eps = 0.1):
        self.A = A if type(A) == torch.Tensor else torch.tensor(A, dtype=torch.float32)  # (n,d)
        self.b = b if type(b) == torch.Tensor else torch.tensor(b, dtype=torch.float32)  # (n,)
        n = self.A.shape[0]
        self.sigma_eps = sigma_eps  # variance of the noise
        import torch.distributions as dist
        self.epsilon = dist.MultivariateNormal(torch.zeros(n), torch.eye(n) * sigma_eps**2)  # noise distribution
        if init_h is None:
            init_h = self.estimate_h(self.A, self.b)
        init_h = torch.tensor(init_h, dtype=torch.float32) if type(init_h) != torch.Tensor else init_h
        if init_h.shape != torch.Size([self.A.shape[1]]):  # if init_h is a scalar, then make it a tensor (d,)
            init_h = torch.ones(self.A.shape[1]) * init_h
        self.init_h = init_h
        # prior distribution of h
        self.p_mu_h = torch.tensor(init_h, dtype=torch.float32)  # prior mean of h. (d,)
        self.p_sigma_h = p_sigma_h if type(p_sigma_h) == torch.Tensor else torch.tensor(p_sigma_h, dtype=torch.float32) # prior variance of h. (d,) or a scalar


    @staticmethod
    def estimate_h(A, b):
        '''
        Given A(n,m) and b(n,). Find mu (1,) such that A*mu = b
        Simply calculate the average of the values of h
        :param A:
        :param b:
        :return:
        '''
        rowsumA = torch.sum(A, dim=1)
        h = b/rowsumA  # (n,)
        h = torch.mean(h)  # (1,)
        return h # (1,) --> estimate h based on the average distance across the gene

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
        nll = -self.epsilon.log_prob(y_pred - self.b).mean()
        # calculate the KL divergence between q(h)~lognormal(q_mu_h, q_sigma_h) and p(h)~lognormal(p_mu_h, p_sigma_h)
        d = q_mu_h.shape[0]
        kl_divergence = d*0.5 * (torch.log(self.p_sigma_h**2)-torch.log(q_sigma_h**2) + (q_sigma_h**2/self.p_sigma_h**2) - 1) + \
              0.5 * (1/q_sigma_h**2) * torch.sum((q_mu_h-self.p_mu_h)**2)
        return nll + kl_divergence
    def solve(self, num_epoch = 1000):
        '''
        Solve for the system of linear equations A/h = b
        :return: h
        '''
        model = bayesian_VI_Ahb(self.A, self.b, self.init_h, self.p_sigma_h)
        clip_value = 1
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        scheduler = CosineAnnealingLR(optimizer, T_max=100)
        for i in range(num_epoch):
            model.train()
            optimizer.zero_grad()
            y_pred, q_mu_h, q_sigma_h = model.forward(self.A)
            loss = self.elbo_loss(y_pred, q_mu_h, q_sigma_h)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value) #gradient clipping
            optimizer.step()
            if i % 100 == 0:
                print(f'Epoch {i}/{num_epoch} - Loss: {loss.item()}')
            scheduler.step()  # update the learning rate
        model.eval()
        pred_h = model.reparametrize_log_normal(q_mu_h, q_sigma_h)
        return pred_h.detach().numpy(), model.q_mu_h, model.q_sigma_h
