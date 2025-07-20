import torch
import torch.nn as nn
from .simpleModel import simpleModel
from .bayesianModel import bayesianNormal, bayesian_RBF, bayesian_logNormal
from torch.optim.lr_scheduler import CosineAnnealingLR
import time
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using {device} device')
class BaseModelTrainer:
    def __init__(self, d, dataloader, max_time_seconds=10*60):
        """
        Parent class for training models.

        Args:
            model (nn.Module): The PyTorch model to train.
            dataloader (DataLoader): DataLoader for training data.
            optimizer (torch.optim.Optimizer): Optimizer for training the model.
            num_epochs (int): Number of training epochs.
        """
        self.d = d
        self.dataloader = dataloader
        self.max_time_seconds = max_time_seconds

    def loss(self, outputs, targets):
        """
        Placeholder for the loss function.
        Child classes will implement this with their specific loss functions.

        Args:
            outputs (torch.Tensor): Model outputs.
            targets (torch.Tensor): Ground truth labels.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        pass

    def solve(self, lr, num_epochs, *args, **kwargs):
        """
        Training loop implementation.
        """
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        scheduler = CosineAnnealingLR(optimizer, T_max=kwargs.get('T_max', 10))
        self.model.to(device)
        start_time = time.time()
        for epoch in (range(num_epochs)):
            elapsed_time = time.time() - start_time
            if elapsed_time > self.max_time_seconds:
                return  # signal that the time has run out, the child method will handle the returning of the last h
            self.model.train()
            running_loss = 0.0
            for batch_idx, (A, b) in enumerate(self.dataloader):
                A, b = A.to(device), b.to(device)
                optimizer.zero_grad()
                pred_b = self.model(A)
                loss = self.loss(pred_b, b, *args)
                loss.backward()  # Backpropagate the loss
                optimizer.step()  # Update h
                running_loss += loss.item()
            if epoch % 500 == 0:
                avg_loss = running_loss / len(self.dataloader)
                print(
                    f'Epoch [{epoch + 1}/{num_epochs}], Average Loss: {avg_loss:.4f}, Learning Rate: {scheduler.get_last_lr()[0]:.6f}')
            scheduler.step()  # update the learning rate based on the scheduler
        return

class SimpleSolver(BaseModelTrainer):
    def __init__(self, d, init_h=1, dataloader=None, h_min= 0.1, h_max= 5):
        super(SimpleSolver, self).__init__(d, dataloader)
        self.model = simpleModel(d, init_h=init_h).to(device)
        self.h_min = h_min
        self.h_max = h_max

    def loss(self, outputs, targets):
        return (torch.nn.functional.mse_loss(outputs, targets) +
                torch.nn.functional.relu(self.model.h - self.h_max).sum() +
                torch.nn.functional.relu(self.h_min - self.model.h).sum())

    def solve(self, lr=0.01, num_epochs=1000):
        '''
        Solve for the system of linear equations Ax = b
        :return: x
        '''
        # first call on the parent class solve function
        super(SimpleSolver, self).solve(lr, num_epochs)
        # return the last h
        return self.model.h.detach()

class SimpleSmoothSolver(BaseModelTrainer):
    def __init__(self, d, init_h=1, dataloader=None, h_min= 0.1, h_max= 5, lambda_smooth=0.1):
        super(SimpleSmoothSolver, self).__init__(d, dataloader)
        self.model = simpleModel(d, init_h=init_h).to(device)
        self.h_min = h_min
        self.h_max = h_max
        self.lambda_smooth = lambda_smooth

    def loss(self, outputs, targets, avg_h = None):
        smooth_l2_loss = torch.nn.functional.mse_loss(self.model.h[1:], self.model.h[:-1])
        l2_loss_from_avg = torch.nn.functional.mse_loss(self.model.h, avg_h) if avg_h is not None else 0
        return (torch.nn.functional.mse_loss(outputs, targets) +
                self.lambda_smooth * smooth_l2_loss +
                self.lambda_smooth * l2_loss_from_avg +
                torch.nn.functional.relu(self.model.h - self.h_max).sum() +
                torch.nn.functional.relu(self.h_min - self.model.h).sum())

    def solve(self, lr=0.01, num_epochs=1000, avg_h = None):
        '''
        Solve for the system of linear equations Ax = b
        :return: x
        '''
        avg_h = avg_h.to(device)
        avg_h = avg_h.expand_as(self.model.h)
        # first call on the parent class solve function
        super(SimpleSmoothSolver, self).solve(lr, num_epochs, avg_h=avg_h)
        # return the last h
        return self.model.h.detach()

class BayesianNormalSolver(BaseModelTrainer):
    def __init__(self, d, init_h=1, dataloader=None, sigma_ep=0.1):
        super(BayesianNormalSolver, self).__init__(d, dataloader)
        self.model = bayesianNormal(d, init_h=init_h).to(device)
        self.sigma_ep = sigma_ep

    def loss(self, outputs, targets):
        q_mu_h = self.model.q_mu_h
        q_sigma_h = self.model.q_sigma_h
        p_mu_h = self.model.p_mu_h
        p_sigma_h = self.model.p_sigma_h
        n = outputs.shape[0]
        eps_mean = torch.zeros(n, device=outputs.device)
        eps_covar = torch.eye(n, device=outputs.device) * self.sigma_ep
        epsilon = torch.distributions.MultivariateNormal(loc=eps_mean, covariance_matrix=eps_covar)
        nll = -epsilon.log_prob(outputs - targets).mean()
        kl_divergence = self.d * 0.5 * (torch.log(p_sigma_h ** 2) - torch.log(q_sigma_h ** 2) + (q_sigma_h ** 2 / p_sigma_h ** 2) - 1) + \
                        0.5 * (1 / q_sigma_h ** 2) * torch.sum((q_mu_h - p_mu_h) ** 2)
        return nll+kl_divergence

    def solve(self, lr = 0.01, num_epochs = 100):
        super(BayesianNormalSolver, self).solve(lr, num_epochs, T_max=200)
        # return the last h
        return self.model.q_mu_h.detach()


class BayesianRBFSolver(BaseModelTrainer):
    def __init__(self, d, init_h = 1, dataloader=None, sigma_ep=0.1, coords = None, rbf_sigma=1):
        super(BayesianRBFSolver, self).__init__(d, dataloader)
        self.model = bayesian_RBF(d, init_h=init_h, coords=coords, rbf_sigma=rbf_sigma).to(device)
        self.sigma_ep = sigma_ep

    def loss(self, outputs, targets):
        q_mu_h = self.model.q_mu_h
        p_mu_h = self.model.p_mu_h
        Sigma_RBF = self.model.Sigma_RBF
        n = outputs.shape[0]
        eps_mean = torch.zeros(n, device=outputs.device)
        eps_var = torch.eye(n, device=outputs.device) * self.sigma_ep
        epsilon = torch.distributions.MultivariateNormal(loc=eps_mean, covariance_matrix=eps_var)
        nll = -epsilon.log_prob(outputs - targets).mean()
        mu_diff = q_mu_h - p_mu_h
        Sigma_inv_mu_diff = Sigma_RBF.solve(mu_diff)  # Sigma_rbf^-1 * mu_diff
        kl_divergence = 0.5*torch.matmul(mu_diff, Sigma_inv_mu_diff)
        # if X1 ~ Normal(\mu1, \Sigma1) and X2 ~ Normal(\mu2, \Sigma2), then the KL divergence between X1 and X2 is
        # 0.5 * (tr(\Sigma2^-1 \Sigma1) + (\mu2 - \mu1)^T \Sigma2^-1 (\mu2 - \mu1) - k + log(|\Sigma2|/|\Sigma1|))
        # but when \Sigma1 = \Sigma2 = \Sigma, then the KL divergence simplifies to
        # 0.5 * (\mu1-\mu2)^T * \Sigma^-1 * (\mu1-\mu2)
        return nll+kl_divergence

    def solve(self, lr = 0.01, num_epochs = 100):
        super(BayesianRBFSolver, self).solve(lr, num_epochs, T_max=10)
        # return the last h
        return self.model.q_mu_h.detach()

class BayesianLogNormalSolver(BaseModelTrainer):
    def __init__(self, d, init_h=1, dataloader=None, sigma_ep=0.1):
        super(BayesianLogNormalSolver, self).__init__(d, dataloader)
        self.model = bayesian_logNormal(d, init_h=init_h).to(device)
        self.sigma_ep = sigma_ep

    def loss(self, outputs, targets):
        q_mu_h = self.model.q_mu_h
        q_sigma_h = self.model.q_sigma_h
        p_mu_h = self.model.p_mu_h
        p_sigma_h = self.model.p_sigma_h
        n = outputs.shape[0]
        ep_mean = torch.zeros(n, device=outputs.device)
        ep_var = torch.eye(n, device=outputs.device) * self.sigma_ep
        epsilon = torch.distributions.MultivariateNormal(loc=ep_mean, covariance_matrix=ep_var)
        nll = -epsilon.log_prob(outputs - targets).mean()
        kl_divergence = self.d * 0.5 * (torch.log(p_sigma_h ** 2) - torch.log(q_sigma_h ** 2) + (q_sigma_h ** 2 / p_sigma_h ** 2) - 1) + 0.5 * (1 / q_sigma_h ** 2) * torch.sum((q_mu_h - p_mu_h) ** 2)
        return nll+kl_divergence

    def solve(self, lr = 0.01, num_epochs = 100):
        super(BayesianLogNormalSolver, self).solve(lr, num_epochs, T_max=200)
        # return the last h
        h = torch.exp(self.model.q_mu_h.detach())  # it could have been exp(q_mu_h + 0.5 * q_sigma_h**2) but we are assuming that q_sigma_h is small
        return h