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

# Hyper-parameters
learning_rate = 1e-4
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

# Load datasets(MNIST)
path = './'
mnist_t = dsets.MNIST(path, download=True, train=True, transform=transforms.ToTensor())
mnist_val = dsets.MNIST(path, download=True, train=False, transform=transforms.ToTensor())

original = util.DataLoader(mnist_t, shuffle=True)
original_val = util.DataLoader(mnist_val, shuffle=True)

class_name = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0']

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        # input_channel, output_channel(# of filters), filter size
        self.conv1 = nn.Conv2d(1, 16, 5, padding=2)
        # pooling (2 x 2) with stride of 2
        self.pool = nn.MaxPool2d(2, stride=2)
        self.conv2 = nn.Conv2d(16, 8, 3, padding=2)
        # Use the formula to get the input size
        self.fc1 = nn.Linear(2 * 2 * 8, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 10)

    def forward(self, x):
        x = self.pool(f.relu(self.conv1(x)))
        x = self.pool(f.relu((self.conv2(x))))
        x = x.view(-1, 2 * 2 * 8)
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

            img, label = target

            for theta in range(65):
                new_img = ndimage.rotate(img, theta*5, reshape=False)
                new_img = torch.from_numpy(new_img)
                new_img = new_img.to(device)
                label = label.to(device)

                # forward
                prediction = network(new_img.view(1, 1, 28, 28))
                loss_train = loss_fn(prediction.view(1, -1), label)

                # backward
                optimizer.zero_grad()
                loss_train.backward()
                optimizer.step()

            img = img.to(device)
            # forward
            prediction = network(img)
            loss_train = loss_fn(prediction.view(1, -1), label)

            # backward
            optimizer.zero_grad()
            loss_train.backward()
            optimizer.step()
            new_img = torch.zeros(1, 28, 28)

            print(f"Epoch: {i}, {n}th data, Training loss {loss_train.item():.4f},")

            torch.cuda.empty_cache()


n_epochs = 1
training_loop(
    n_epoch=n_epochs,
    network=model,
    optim_fn=optimizer,
    loss_fn=loss,
    data_t=original)

with torch.no_grad():
    n_correct = 0
    n_samples = 0
    n_class_correct = [0 for i in range(10)]
    n_class_samples = [0 for i in range(10)]
    for images, labels in original_val:
        plt.imshow(images.view(28, 28, 1))
        plt.show()
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images).to(device)
        # max returns (value ,index)
        _, predicted = torch.max(outputs, 1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()

        theta = np.random.randint(1, 65)
        new_img = torch.zeros(1, 28, 28)
        new_img = ndimage.rotate(images.cpu().view(1, 28, 28), theta * 5, reshape=False)
        new_img = torch.from_numpy(new_img)

        new_img = new_img.to(device)

        outputs = model(new_img.view(1, 1, 28, 28)).to(device)
        # max returns (value ,index)
        _, predicted = torch.max(outputs, 1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()

    acc = 100.0 * n_correct / n_samples
    print(f'Accuracy of the network: {acc} %')

    for i in range(10):
        acc = 100.0 * n_class_correct[i] / n_class_samples[i]
        print(f'Accuracy of {class_name[i]}: {acc} %')
