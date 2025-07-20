import torch
import gpytorch
torch.manual_seed(9999)

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

    def solve(self, lr=0.1, num_epochs=50):
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
        from solver_Ahb.bayesianGPSolver import BayesianLinearRegression_RBF
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

# Sample data
train_x = torch.load('/gladstone/engelhardt/home/hvu/source/RNA_rates/splicingrates/simulations/elong_analysis/A.pt')
train_x = torch.tensor(train_x, dtype=torch.float32)
h_bins = 0.2
coordinates = torch.arange(0, train_x.shape[1]) * h_bins + h_bins / 2
print('coordinates', coordinates)
train_y = 5 * torch.ones(train_x.shape[0])
true_h = 1.65
true_beta = torch.ones(train_x.shape[1])* 0.6009 #(1/true_h)
# Define prior mean and variance for beta
prior_mean = true_beta.clone().detach()  # Let's just give it the true beta value as prior. I want to test if it is performing how I expect it to do

# from solver_Ahb.bayesianGPSolver import bayesianRBF_solver
from solver_Ahb.bayesianGPSolver import BayesianLinearRegression_RBF


# Define the likelihood and model
likelihood = gpytorch.likelihoods.GaussianLikelihood()
model = BayesianLinearRegression_RBF(train_x, train_y, coordinates, prior_mean, likelihood)
for name, param in model.named_parameters():
    print(f"Parameter name: {name}, Shape: {param.shape}")
# Training mode
model.train()
likelihood.train()
# Define optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
num_epochs = 50
for i in range(num_epochs):
    optimizer.zero_grad()
    output = model(train_x)
    loss = -mll(output, train_y)
    loss.backward()
    optimizer.step()
model.eval()
likelihood.eval()
# print(model.mean_module.weights)
print(1/model.mean_module.weights.detach())

B_RBFSolver = bayesianRBF_solver(train_x,train_y, h_bins= h_bins)
# import pdb; pdb.set_trace()
pred_h = B_RBFSolver.solve(num_epochs=num_epochs, lr = 0.1)
print(pred_h)
