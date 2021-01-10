# Deep Learning with Pytorch Ch.5
# Simple Linear Regression
import torch

t1 = [0.5, 14.0, 15.0, 28.0, 11.0, 8.0, 3.0, -4.0, 6.0, 13.0, 21.0]
t2 = [35.7, 55.9, 58.2, 81.9, 56.3, 48.9, 33.9, 21.8, 48.4, 60.4, 68.4]

t1 = torch.tensor(t1)
t2 = torch.tensor(t2)

def model(x, w, b):
    return w * x + b

def loss_fn(p, c):
    sq_diff = (p - c) ** 2
    return sq_diff.mean()


w = torch.ones(())
b = torch.zeros(())

p = model(t1, w, b)
loss = loss_fn(p, t2)

print(loss)

delta = 0.1
learning_rate = 1e-2

loss_rate_w = (loss_fn(model(t2, w + delta, b), t1) - loss_fn(model(t2, w - delta, b), t1)) / (2.0*delta)
w = w - learning_rate * loss_rate_w

loss_rate_b = (loss_fn(model(t2, w, b + delta), t1) - loss_fn(model(t2, w, b - delta), t1)) / (2.0 * delta)
b = b - learning_rate * loss_rate_b


def dloss_fn(p, c):
    dsq_diffs = 2 * (p - c) / p.size(0)
    return dsq_diffs

def dmodel_dw(x, w, b):
    return x

def dmodel_db(x, w, b):
    return 1.0

def grad_fn(t_u, t_c, t_p, w, b):
    dloss_dtp = dloss_fn(t_p, t_c)
    dloss_dw = dloss_dtp * dmodel_dw(t_u, w, b)
    dloss_db = dloss_dtp * dmodel_db(t_u, w, b)
    return torch.stack([dloss_dw.sum(), dloss_db.sum()])

def training_loop(n_epochs, learning_rate, params, t_u, t_c):
    for epoch in range(1, n_epochs + 1):
        w, b = params
        t_p = model(t_u, w, b)
        loss = loss_fn(t_p, t_c)
        grad = grad_fn(t_u, t_c, t_p, w, b)
        params = params - learning_rate * grad  # Update rule
        print('Epoch %d, Loss %f' % (epoch, float(loss)))

        print("\tParams:", params)
        print("\tGrag:  ", grad)
    return params

training_loop(
    n_epochs = 100,
    learning_rate = 1e-4,
    params = torch.tensor([1.0, 0.0]),
    t_u = t2,
    t_c = t1)