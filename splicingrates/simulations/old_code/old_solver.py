import numpy as np
import torch
def solve_Ahinv_b(A, b, num_epochs=5000, lr=0.01, verbose=True):
    '''
    Given the matrix A, the vector b, and the bounds for h_inv, solve the equation Ah_inv = b
    :param A:
    :param b:
    :param bounds:
    :return:
    '''
    # h_inv = find_linreg_slope(A, b)
    # h = 1 / h_inv
    # return h
    import torch
    n = A.shape[0]
    m = A.shape[1]
    h_inv = torch.randn(m, 1, requires_grad=True)
    At = torch.tensor(A).float()  # shape n(samples) * m (bins)
    bt = torch.tensor(b).float().unsqueeze(1)  # shape n*1
    optimizer = torch.optim.Adam([h_inv], lr=lr)
    loss_fn = torch.nn.MSELoss()
    for epoch in range(num_epochs):
        optimizer.zero_grad()  # Zero the gradients
        Ah = torch.mm(At, h_inv)  # Compute Ah
        penalty_min = torch.sum(torch.clamp(0.1 - h_inv, min=0) ** 2)
        penalty_max = torch.sum(torch.clamp(h_inv - 10, min=0) ** 2)
        loss = loss_fn(Ah, bt) + penalty_min + penalty_max  # Compute the loss
        loss.backward()  # Backpropagate the loss
        optimizer.step()  # Update h
        if epoch % 500 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")
    h = 1 / h_inv.detach().numpy().flatten()
    return h
