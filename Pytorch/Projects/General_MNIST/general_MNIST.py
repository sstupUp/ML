import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as dsets
import torch.utils.data as util
import torchvision.transforms as transforms
import torch.nn.functional as f
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
import tqdm

# Hyper-parameters
batch_size = 64
learning_rate = 1e-4
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

# Load datasets(MNIST)
path = './'
mnist_t = dsets.MNIST(path, download=True, train=True, transform=transforms.ToTensor())
mnist_val = dsets.MNIST(path, download=True, train=False, transform=transforms.ToTensor())

original = util.DataLoader(mnist_t, shuffle=True, batch_size=batch_size)
original_val = util.DataLoader(mnist_val, shuffle=True, batch_size=batch_size)

class_name = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0']

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3, 3), stride=(2, 2), padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(5, 5), stride=(2, 2), padding=2)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(2 * 2 * 64, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 10)

    def forward(self, x):
        x = self.pool(f.relu(self.conv1(x)))
        # Batch Normalization(x)
        x = self.pool(f.relu(self.conv2(x)))
        # Batch Normalization(x)

        # Flattening
        x = x.view(x.size(0), -1)
        x = f.relu(self.fc1(x))
        x = f.relu(self.fc2(x))
        x = self.fc3(x)

        return x


# Optimizer & loss
model = ConvNet().to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
loss = nn.CrossEntropyLoss()


def training_loop(n_epoch, network, optim_fn, loss_fn, data_t):
    for i in range(0, n_epoch + 1):
        for n, target in enumerate(data_t):
            if n == 600:
                break
            img, label = target
            for theta in range(65):
                new_img = ndimage.rotate(img, theta*5, reshape=False)
                new_img = torch.from_numpy(new_img)
                new_img = new_img.to(device)
                label = label.to(device)

                # forward
                prediction = network(new_img)
                loss_train = loss_fn(prediction, label)

                # backward
                optimizer.zero_grad()
                loss_train.backward()
                optimizer.step()


                print(f"Epoch: {i}, {n}th data, Training loss {loss_train.item():.4f},")




n_epochs = 0
training_loop(
    n_epoch=n_epochs,
    network=model,
    optim_fn=optimizer,
    loss_fn=loss,
    data_t=original)

def accuracy(model, data, classes):
    print('Calculating Accuracy...')
    with torch.no_grad():
        n_correct = 0
        n_samples = 0
        acc_arr = [0 for i in range(classes)]
        acc = 0

        n_class_correct = [0 for i in range(10)]
        n_class_samples = [0 for i in range(10)]

        for target in tqdm.tqdm(data):
            images, labels = target
            images = images.view(images.size(0), 1, 28, 28).to(device)

            labels = labels.to(device)
            outputs = model(images).to(device)
            # max returns (value ,index)
            _, predicted = torch.max(outputs, 1)

            n_samples += len(labels)
            n_correct += (predicted == labels).sum().item()

            for i in range(images.size(0)):
                label = labels[i]
                pred = predicted[i]

                if (label == pred):
                    n_class_correct[label] += 1
                    n_class_samples[label] += 1

        acc = 100.0 * n_correct / n_samples
        for i in range(10):
            acc_arr[i] = 100.0 * n_class_correct[i] / n_class_samples[i]

        return acc, acc_arr

tot_acc, idv_acc = accuracy(model, original_val, 10)
print(f'Accuracy of the network: {tot_acc} %')
for i in range(10):
    print(f'Accuracy of {class_name[i]}: {idv_acc[i]} %')