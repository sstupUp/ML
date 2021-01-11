# Simple Linear Regression using Autograd mechanism
# The upgraded version of 'Simple_LR.py'
import torch

# Data Set
# We want to predict t_c values using t_u
t_c = [0.5, 14.0, 15.0, 28.0, 11.0, 8.0, 3.0, -4.0, 6.0, 13.0, 21.0]
t_u = [35.7, 55.9, 58.2, 81.9, 56.3, 48.9, 33.9, 21.8, 48.4, 60.4, 68.4]

t_c = torch.tensor(t_c)
t_u = torch.tensor(t_u)

params = torch.tensor([1.0, 0.0], requires_grad=True)

def model(x, w, b):
    return w * x + b

def loss_fn(t_p, t_c):
    diff = (t_p - t_c) ** 2
    return diff.mean()

def training_loop(n_epochs, learning_rate, params, t_u, t_c):
    for epoch in range(1, n_epochs + 1):

        # Since .backward() is ACCUMULATED, we need to zero them out before each iteration
        if params.grad is not None:
            params.grad.zero_()

        t_p = model(t_u, *params)
        loss = loss_fn(t_p, t_c)

        # Calculate the grad of w, b
        # Pytorch can automatically calculate them
        loss.backward()

        with torch.no_grad():
            params -= learning_rate * params.grad

        if epoch % 500 == 0:
            print('Epoch %d, Loss %f' % (epoch, float(loss)))

    return params

t_un = t_u * 0.1

params = training_loop(
n_epochs = 5000,
learning_rate = 1e-2,
params = torch.tensor([1.0, 0.0], requires_grad=True),
t_u = t_un,
t_c = t_c)

print(params)