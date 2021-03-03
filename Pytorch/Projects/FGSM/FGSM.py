# Fast Gradient Sign Method using simple CNN and MNIST
#
#
# https://www.pyimagesearch.com/2021/03/01/adversarial-attacks-with-fgsm-fast-gradient-sign-method/
# The above blog used Keras & TensorFlow

import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torch.utils.data as utils
import torch.optim as optim
import torchvision.transforms as transforms
import torch.nn.functional as f
import numpy as np
import matplotlib.pyplot as plt

# Hyper-parameters
learning_rate = 1e-5
batch_size = 128

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

# Load datasets(MNIST)
path = './'
MNIST_t = dsets.MNIST(path, download=True, train=True, transform=transforms.ToTensor())
MNIST_val = dsets.MNIST(path, download=True, train=False, transform=transforms.ToTensor())

data_t = utils.DataLoader(MNIST_t, shuffle=True, batch_size=batch_size)
data_val = utils.DataLoader(MNIST_val, shuffle=True, batch_size=batch_size)

class_name = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']


# Network
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()

        # 1st conv layer = 32 3*3 filters, with 2*2 stride
        # CNN => ReLU => Batch Normalization(which is skipped in this code)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3, 3), stride=(2, 2))
        # No padding
        # No Pooling

        # 2nd conv layer = 64 3*3 filters, with 2*2 stride
        # CNN => ReLU => Batch Normalization(which is skipped in this code)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=(2, 2))
        # No padding
        # No Pooling

        # 1st and only FC
        # FC => ReLU (=> Batch Normalization)=> Dropout(0.5)
        # Since the data is MNIST, the model has to output 10 values
        # Linear( W_out * H_out * N_channel, N_classes)
        self.fc = nn.Linear(6 * 6 * 64, 10)

        # The blog used MSE and Softmax as loss function
        # In this code, I will use CrossEntropyLoss to do them at the same time

    def forward(self, x):
        x = f.relu(self.conv1(x))
        # Batch Normalization(x)
        x = f.relu((self.conv2(x)))
        # Batch Normalization(x)

        # Flattening
        x = x.view(-1, 6 * 6 * 64)
        # default drop-out probability = 0.5
        x = f.dropout(f.relu(self.fc(x)))

        return x


# Optimizer & loss_fn
model = ConvNet().to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
loss = nn.CrossEntropyLoss()


# Training loop
def training_loop(n_epoch, network, optim_fn, loss_fn, data_t):
    for i in range(0, n_epoch):
        for n, target in enumerate(data_t):
            img, label = target
            img = img.to(device)
            label = label.to(device)
            # forward
            prediction = network(img)
            loss_train = loss_fn(prediction, label)

            # backward
            optimizer.zero_grad()
            loss_train.backward()
            optimizer.step()

            print(f"Epoch: {i + 1}, Batch: {n * batch_size}/60000, Training loss {loss_train.item():.4f},")


# We pass an image through the trained-model, and calculate the loss of the image
# Then, we use the sign of the gradiant of loss to generate adversarial attack
def generate_image_adversary(model, loss_fn, data, batch_size, eps=2/255.0):
    for img, label in data:
        img = img.to(device)
        label = label.to(device)




print('Training the model')

# run
n_epochs = 5
training_loop(
    n_epoch=n_epochs,
    network=model,
    optim_fn=optimizer,
    loss_fn=loss,
    data_t=data_t)

print('Calculating Accuracy...')

with torch.no_grad():
    n_correct = 0
    n_samples = 0
    n_class_correct = [0 for i in range(10)]
    n_class_samples = [0 for i in range(10)]
    i = 0
    for images, labels in data_val:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images).to(device)
        # max returns (value ,index)
        _, predicted = torch.max(outputs, 1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()

        if len(images) != batch_size:
            break

        for i in range(batch_size):
            label = labels[i]
            pred = predicted[i]

            if (label == pred):
                n_class_correct[label] += 1
            n_class_samples[label] += 1

    acc = 100.0 * n_correct / n_samples
    print(f'Accuracy of the network: {acc} %')

    for i in range(10):
        acc = 100.0 * n_class_correct[i] / n_class_samples[i]
        print(f'Accuracy of {class_name[i]}: {acc} %')

with torch.no_grad():
    for k in range(5):
        tmp = data_val.dataset[k][0]
        target = data_val.dataset[k][1]
        plt.imshow(tmp.view(28, 28))
        plt.show()
        tmp = tmp.view(1, 1, 28, 28)
        tmp = tmp.to(device)
        outputs = model(tmp).to(device)
        # max returns (value ,index)
        _, predicted = torch.max(outputs, 1)
        print('target', target)
        print(class_name[predicted[0]])
