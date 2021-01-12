# Simple Linear Regression using Autograd & optimizer mechanism
# The upgraded version of 'SLR_Backward.py'
import torch
# Use 'dir(optim)' to check the list of the optimizers that Pytorch provides
import torch.optim as optim

# Data Set
t_c = [0.5, 14.0, 15.0, 28.0, 11.0, 8.0, 3.0, -4.0, 6.0, 13.0, 21.0]
t_u = [35.7, 55.9, 58.2, 81.9, 56.3, 48.9, 33.9, 21.8, 48.4, 60.4, 68.4]

t_c = torch.tensor(t_c)
t_u = torch.tensor(t_u)
# Normalizing the input
t_un = t_u * 0.1

# Setting the parameters: w, b
params = torch.tensor([1.0, 0.0], requires_grad=True)
learning_rate = 1e-02
# Making optimizer which will update the parameters and set the grads to zero
# SGD stands for Stochastic Gradient Descent
optimizer = optim.SGD([params], lr=learning_rate)

def model(x, w, b):
    return w * x + b

def loss_fn(t_p, t_c):
    diff = (t_p - t_c) ** 2
    return diff.mean()

def training_loop(n_epochs, optimizer, params, t_u, t_c):
    for epoch in range(1, n_epochs + 1):

        t_p = model(t_u, *params)
        loss = loss_fn(t_p, t_c)

        # Set the grads of params before calling .backward()
        optimizer.zero_grad()
        # Calculating the grads of params
        loss.backward()
        # Updating params
        optimizer.step()

        if epoch % 500 == 0:
            print('Epoch %d, Loss %f' % (epoch, float(loss)))

    return params

params = training_loop(
    n_epochs = 5000,
    optimizer=optimizer,
    params = params,
    t_u = t_un,
    t_c = t_c)

print('SGD:', params, '\n')

# Using another optimizer(Adam)
params2 = torch.tensor([1.0, 0.0], requires_grad=True)
learning_rate2 = 1e-1
optim2 = optim.Adam([params2], lr=learning_rate2)

params2 = training_loop(
    n_epochs=2000,
    optimizer=optim2,
    params=params2,
    t_u=t_u,
    t_c=t_c)
print('Adam:', params2)

