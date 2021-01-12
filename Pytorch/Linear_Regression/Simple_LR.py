# Deep Learning with Pytorch Ch.5
# Simple Linear Regression
import torch
from matplotlib import pyplot as plt


# Data Set
# We want to predict t_c values using t_u
t_c = [0.5, 14.0, 15.0, 28.0, 11.0, 8.0, 3.0, -4.0, 6.0, 13.0, 21.0]
t_u = [35.7, 55.9, 58.2, 81.9, 56.3, 48.9, 33.9, 21.8, 48.4, 60.4, 68.4]

t_c = torch.tensor(t_c)
t_u = torch.tensor(t_u)

# Simple LR model
def model(x, w, b):
    return w * x + b

# MSE loss function
def loss_fn(t_p, t_c):
    sq_diff = (t_p - t_c) ** 2
    return sq_diff.mean()

# Random Weights and Bias
w = torch.ones(())
b = torch.zeros(())

# defs for gradiant
def dloss_fn(t_p, t_c):
    dsq_diffs = 2 * (t_p - t_c) / t_p.size(0)
    return dsq_diffs

def dmodel_dw(x, w, b):
    return x

def dmodel_db(x, w, b):
    return 1.0

# Actual gradiant function
def grad_fn(t_u, t_c, t_p, w, b):
    dloss_dtp = dloss_fn(t_p, t_c)
    dloss_dw = dloss_dtp * dmodel_dw(t_u, w, b)
    dloss_db = dloss_dtp * dmodel_db(t_u, w, b)
    return torch.stack([dloss_dw.sum(), dloss_db.sum()])

# Training Function
def training_loop(n_epochs, learning_rate, params, t_u, t_c, print_epoch=True, print_para=True):
    for epoch in range(1, n_epochs + 1):
        w, b = params
        t_p = model(t_u, w, b)
        loss = loss_fn(t_p, t_c)
        grad = grad_fn(t_u, t_c, t_p, w, b)
        params = params - learning_rate * grad  # Update rule

        if print_epoch:
          print('Epoch %d, Loss %f' % (epoch, float(loss)))

        if print_para:
          print("\tParams:", params)
          print("\tGrag:  ", grad)

    if ~print_epoch & ~print_para:
        print('Epoch %d, Loss %f' % (epoch, float(loss)))
        print("\tParams:", params)
        print("\tGrag:  ", grad)

    return params

# Normalizing the input
t_un = 0.1 * t_u

# Training the model
params = training_loop(
    n_epochs = 100,
    learning_rate = 1e-2,
    params = torch.tensor([1.0, 0.0]),
    t_u = t_u,
    t_c = t_c,
    print_epoch=False,
    print_para=True)

'''
# Plotting the model
t_p = model(t_un, *params)
fig = plt.figure(dpi=120)
plt.xlabel("Temperature (°Fahrenheit)")
plt.ylabel("Temperature (°Celsius)")
plt.plot(t_u.numpy(), t_p.detach().numpy())
plt.plot(t_u.numpy(), t_c.numpy(), 'o')
plt.show()
'''