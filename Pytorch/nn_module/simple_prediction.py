import torch
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

data_path = './student-mat.csv'
dataset = pd.read_csv(data_path, delimiter=';')

num_col = ['age', 'absences', 'freetime', 'G1', 'G2']
label_col = ['G3']

d = dataset.loc[1:, num_col]
label = dataset.loc[1:, label_col]
d = torch.tensor(d.to_numpy(), dtype=torch.float32)
d = torch.reshape(d, (-1, 5))

label = torch.tensor(label.to_numpy(), dtype=torch.float32)
label = torch.reshape(label, (-1, 1))

def randomidx(n, idx_pct=0.1):
    temp = np.random.permutation(n)
    print(n)
    val_idx = int(n * idx_pct)
    print(val_idx)
    return temp[:val_idx], temp[val_idx:]

val_idx, test_idx = randomidx(len(dataset) - 1, idx_pct=0.2)
d_test = d[test_idx,:]
test_label = label[test_idx,:]
d_val = d[val_idx,:]
val_label = label[val_idx,:]

print(d_test.shape, test_label.shape, d_val.shape, val_label.shape)

sequ_model = nn.Sequential(
    nn.Linear(5, 64),
    nn.Tanh(),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 1)
)
linear_model = nn.Linear(5, 1)

learning_rate = 1e-4
linear_rate = 1e-4
optimizer = optim.SGD(sequ_model.parameters(), lr=learning_rate)
linear_optimizer = optim.SGD(linear_model.parameters(), lr=linear_rate)
loss_fn = nn.MSELoss()

def training_loop(n_epochs, model, optim, loss_fn, d_t, test_label, d_val, val_label):
    for epoch in range(1, n_epochs + 1):
        pred = model(d_t)
        loss = loss_fn(pred, test_label)

        with torch.no_grad():
            val_pred = model(d_val)
            val_loss = loss_fn(val_pred, val_label)
            assert val_loss.requires_grad == False

        optim.zero_grad()
        loss.backward()
        optim.step()

        if epoch == 1 or epoch % 100 == 0:
            print(f"Epoch {epoch}, Training loss {loss.item():.4f},"
                  f" Validation loss {val_loss.item():.4f}")

loss = training_loop(
    n_epochs=3000,
    model=sequ_model,
    optim=optimizer,
    loss_fn=loss_fn,
    d_t=d_test,
    test_label=test_label,
    d_val=d_val,
    val_label=val_label
)

print('Linear Regression')

loss2 = training_loop(
    n_epochs=2000,
    model=linear_model,
    optim=linear_optimizer,
    loss_fn=loss_fn,
    d_t=d_test,
    test_label=test_label,
    d_val=d_val,
    val_label=val_label
)

# Plotting the data
print(d_test[:, 4], test_label.shape)
plt.scatter(d_test[:, 4], test_label)
plt.xlabel('G2')
plt.ylabel('Filnal Grage')
plt.show()