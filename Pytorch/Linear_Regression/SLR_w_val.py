# We separate the input into the training data and validation data
# This file is the upgraded version of the SLR_Backward_w_optim
import torch
import torch.optim as optim

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

# Setting the parameters: w, b
params = torch.tensor([1.0, 0.0], requires_grad=True)
learning_rate = 1e-02
optimizer = optim.SGD([params], lr=learning_rate)

def model(x, w, b):
    return w * x + b

def loss_fn(t_p, t_c):
    diff = (t_p - t_c) ** 2
    return diff.mean()

def training_loop(n_epochs, optimizer, params, train_t_u, val_t_u, train_t_c, val_t_c):
    for epoch in range(1, n_epochs + 1):

        train_t_p = model(train_t_u, *params)
        train_loss = loss_fn(train_t_p, train_t_c)

        # Use .no_grad() not to create the computation tree of val_data
        with torch.no_grad():
            val_t_p = model(val_t_u, *params)
            val_loss = loss_fn(val_t_p, val_t_c)
            assert val_loss.requires_grad == False

        # Set the grads of params before calling .backward()
        optimizer.zero_grad()
        # Calculating the grads of params
        # We don't call val_loss.backward(), since we are not fitting the model based on valid data
        train_loss.backward()
        # Updating params
        optimizer.step()

        if epoch <= 3 or epoch % 500 == 0:
            print(f"Epoch {epoch}, Training loss {train_loss.item():.4f},"
                  f" Validation loss {val_loss.item():.4f}")

    return params

params = training_loop(
    n_epochs = 3000,
    optimizer=optimizer,
    params= params,
    train_t_u= train_t_un,
    val_t_u= val_t_un,
    train_t_c= train_t_c,
    val_t_c= val_t_c)

print('SGD:', params, '\n')
