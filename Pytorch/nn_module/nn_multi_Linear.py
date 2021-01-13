import torch
import torch.nn as nn
import torch.optim as optim
from collections import OrderedDict
from matplotlib import pyplot as plt

# Data Set
t_c = [0.5, 14.0, 15.0, 28.0, 11.0, 8.0, 3.0, -4.0, 6.0, 13.0, 21.0]
t_u = [35.7, 55.9, 58.2, 81.9, 56.3, 48.9, 33.9, 21.8, 48.4, 60.4, 68.4]

t_c = torch.tensor(t_c)
t_u = torch.tensor(t_u)

n_samples = t_u.shape[0]
# We will use 20% of the input as the validation set
n_val = int(0.2 * n_samples)

# Creating random index for the input data
shuffled_index = torch.randperm(n_samples)
train_index = shuffled_index[:-n_val]
val_index = shuffled_index[-n_val:]

# Training set
train_t_u = t_u[train_index]
train_t_c = t_c[train_index]

# Validation set
val_t_u = t_u[val_index]
val_t_c = t_c[val_index]

train_t_un = 0.1 * train_t_u
val_t_un = 0.1 * val_t_u

# Making the data into one feature form
train_t_c = torch.unsqueeze(train_t_c, 1)
train_t_un = torch.unsqueeze(train_t_un, 1)
val_t_c = torch.unsqueeze(val_t_c, 1)
val_t_un = torch.unsqueeze(val_t_un, 1)

# .parameters() will give us the weight and bias from the 1st and 2nd Linear modules
# for param in seq_model.parameters():
#    print(param.shape)

# parameters in nn.Modules can be identified by name
# for name, param in seq_model.named_parameters():
#    print(name, param.shape)

# nn.Sequencial accepts an OrderedDict, in which we can name each module passed to Sequencial
seq_model = nn.Sequential(OrderedDict([
    ('input_linear', nn.Linear(1, 8)),
    ('hidden_activation', nn.Tanh()),
    ('output_linear', nn.Linear(8, 1))
    ]))
# print(seq_model)
# It gets more explanatory names for submodules
# for name, param in seq_model.named_parameters():
#     print(name, param.shape)

optimizer = optim.SGD(seq_model.parameters(), lr=1e-3)

def training_loop(n_epochs, optimizer, model, loss_fn, t_u_train, t_u_val, t_c_train, t_c_val):
    for epoch in range(1, n_epochs + 1):
        t_p_train = model(t_u_train)
        loss_train = loss_fn(t_p_train, t_c_train)

        t_p_val = model(t_u_val)

        loss_val = loss_fn(t_p_val, t_c_val)

        optimizer.zero_grad()
        loss_train.backward()
        optimizer.step()

        if epoch == 1 or epoch % 1000 == 0:
            print(f"Epoch {epoch}, Training loss {loss_train.item():.4f},"
                  f" Validation loss {loss_val.item():.4f}")

training_loop(
    n_epochs=3000,
    optimizer= optimizer,
    model= seq_model,
    loss_fn= nn.MSELoss(),
    t_u_train=train_t_un,
    t_u_val=val_t_un,
    t_c_train=train_t_c,
    t_c_val=val_t_c)

print('output:\n', seq_model(val_t_un))
print('answer:\n', val_t_c)
print('input linear grads:\n', seq_model.input_linear.weight.grad)

# Plotting the model
# As we can see, the data was supposed to be linear(Celsius/Fahrenheit relation)
# but the nn tried to fit the noisy data,too. The model is overfed
# o is the input, x is the model output
t_range = torch.arange(20., 90.).unsqueeze(1)
fig = plt.figure(dpi=120)
plt.xlabel("Fahrenheit")
plt.ylabel("Celsius")
plt.plot(t_u.numpy(), t_c.numpy(), 'o')
plt.plot(t_range.numpy(), seq_model(0.1 * t_range).detach().numpy(), 'c-')
temp = 0.1*t_u
temp = torch.unsqueeze(temp, 1)
plt.plot(t_u.numpy(), seq_model(temp).detach().numpy(), 'kx')
plt.show()
