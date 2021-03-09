import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torch.utils.data as utils
import torch.optim as optim
import torchvision.transforms as transforms
import torch.nn.functional as f
import numpy as np
import matplotlib.pyplot as plt
import random
import tqdm

# //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

# Hyper-parameters
learning_rate = 1e-4
batch_size = 128
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
eps = 0.3

# Load datasets(MNIST)
path = 'C:\\Users\\smu\\PycharmProjects\\An\\Machine_Learning\\Pytorch\\Projects\\FGSM'
MNIST_t = dsets.MNIST(path, download=True, train=True, transform=transforms.ToTensor())
MNIST_val = dsets.MNIST(path, download=True, train=False, transform=transforms.ToTensor())

data_t = utils.DataLoader(MNIST_t, shuffle=True, batch_size=batch_size, )
data_val = utils.DataLoader(MNIST_val, shuffle=True, batch_size=batch_size, drop_last=True)

class_name = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']


# //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

# Network
class AutoEncoder(nn.Module):

    def __init__(self):
        in_size = 28 * 28
        medium_size = 300
        code_size = 150

        super(AutoEncoder, self).__init__()
        self.fc = nn.Linear(in_size, medium_size)
        self.code = nn.Linear(medium_size, code_size)
        self.fc_end = nn.Linear(code_size, in_size)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = self.fc(x)
        x = self.code(x)
        x = self.fc_end(x)

        return x


# //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

# Optimizer & loss_fn
model = AutoEncoder().to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
loss = nn.MSELoss()


def training_loop(n_epoch, network, optim_fn, loss_fn, data_t):
    for i in range(0, n_epoch):
        loss_sum = 0
        for target in tqdm.tqdm(data_t):
            img, label = target
            img = img.to(device)

            # forward
            prediction = network(img)

            loss_train = loss_fn(prediction, img.view(img.size(0), -1))
            loss_sum += loss_train.item()
            # backward
            optimizer.zero_grad()
            loss_train.backward()
            optimizer.step()

            #print(f"Epoch: {i + 1}, Batch: {n * batch_size}/60000, Training loss {loss_train.item():.4f},")
        print(f'\naverage loss in {i+1}th epoch: {loss_sum/((600000//batch_size)+1)}')

training_loop(1, model, optimizer, loss, data_t)

with torch.no_grad():
    img, label = data_val.dataset[0]
    print(img)

    result = model(img.to(device))
    result = result.to('cpu')


plt.imshow(img.view(28, 28, 1))
plt.show()

plt.imshow(result.view(28, 28, 1))
plt.show()
