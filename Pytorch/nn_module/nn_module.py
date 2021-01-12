# Linear Regression using nn.module
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

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

# Making linear model using nn.module
# Linear( #_of_features, the_size_of_output)
linear_model = nn.Linear(1, 1)
optimizer = optim.SGD(linear_model.parameters(), lr=1e-2)

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
    model= linear_model,
    loss_fn= nn.MSELoss(),
    t_u_train=train_t_un,
    t_u_val=val_t_un,
    t_c_train=train_t_c,
    t_c_val=val_t_c)

print()
print('weight: ', linear_model.weight)
print('bias:', linear_model.bias)

# Plotting the data
data = [0]*100
for i in range(0, 100):
    data[i] = linear_model.weight * i*0.1 + linear_model.bias

plt.xlabel('t_u')
plt.ylabel('t_c')
plt.plot(t_u, t_c, 'o')
plt.plot(data, linewidth=2.5)
plt.show()
