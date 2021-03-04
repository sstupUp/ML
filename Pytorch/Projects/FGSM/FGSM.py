# Fast Gradient Sign Method using simple CNN and MNIST
#
#
# https://www.pyimagesearch.com/2021/03/01/adversarial-attacks-with-fgsm-fast-gradient-sign-method/
# The above blog used Keras & TensorFlow

#//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

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

#//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

# Hyper-parameters
learning_rate = 1e-4
batch_size = 128
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
eps = 0.3

# Load datasets(MNIST)
path = './'
MNIST_t = dsets.MNIST(path, download=True, train=True, transform=transforms.ToTensor())
MNIST_val = dsets.MNIST(path, download=True, train=False, transform=transforms.ToTensor())

data_t = utils.DataLoader(MNIST_t, shuffle=True, batch_size=batch_size)
data_val = utils.DataLoader(MNIST_val, shuffle=True, batch_size=batch_size, drop_last=True)

class_name = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

#//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

# Network
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()

        '''
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
        '''

        # For the sake of studying ML, I will use deeper network
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3,3), stride=(2,2), padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(5,5), stride=(2,2), padding=2)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(2*2*64, 32)
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

#//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

# Optimizer & loss_fn
model = ConvNet().to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
loss = nn.CrossEntropyLoss()

#//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


# Training loop
def training_loop(n_epoch, network, optim_fn, loss_fn, data_t):
    for i in range(0, n_epoch):
        loss_sum = 0
        for target in tqdm.tqdm(data_t):
            img, label = target
            img = img.to(device)
            label = label.to(device)
            # forward
            prediction = network(img)
            loss_train = loss_fn(prediction, label)
            loss_sum += loss_train.item()
            # backward
            optimizer.zero_grad()
            loss_train.backward()
            optimizer.step()

            #print(f"Epoch: {i + 1}, Batch: {n * batch_size}/60000, Training loss {loss_train.item():.4f},")
        print(f'\naverage loss in {i+1}th epoch: {loss_sum/((600000//batch_size)+1)}')

#//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


# We pass an image through the trained-model, and calculate the loss of the image
# Then, we use the sign of the gradiant of loss to generate adversarial attack

# adv=original+ϵ∗sign(∇xJ(x,θ))
def generate_image_adversary(model, loss_fn, data, eps=0.3):

    img, label = data

    img = img.to(device)
    label = label.to(device)

    img.requires_grad = True

    model.zero_grad()
    pred = model(img).to(device)

    loss = loss_fn(pred, label).to(device)
    # we need to calculate ∇xJ(x,θ)
    loss.backward()
    img.requires_grad = False

    tmp = img + eps*img.grad.data.sign()
    tmp = torch.clamp(tmp, 0, 1)

    return tmp

#//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


def accuracy(model, data, classes, attack=False, eps=eps):
    print('Calculating Accuracy...')
    if attack == False:
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
    else:
        n_correct = 0
        n_samples = 0
        acc_arr = [0 for i in range(classes)]
        acc = 0

        n_class_correct = [0 for i in range(10)]
        n_class_samples = [0 for i in range(10)]

        for target in tqdm.tqdm(data):
            images, labels = target
            adv_img = generate_image_adversary(model=model, loss_fn=loss, data=target, eps=eps)

            labels = labels.to(device)
            outputs = model(adv_img).to(device)
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


#//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


print('\nTraining the model')
n_epochs = 10
training_loop(
    n_epoch=n_epochs,
    network=model,
    optim_fn=optimizer,
    loss_fn=loss,
    data_t=data_t)

#//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


tot_acc, idv_acc = accuracy(model, data_val, 10)
print(f'Accuracy of the network: {tot_acc} %')
for i in range(10):
    print(f'Accuracy of {class_name[i]}: {idv_acc[i]} %')

# eps_arr = [0.001, 0.01, 0.1, 0.15, 0.2, 0.3]

tot_acc, idv_acc = accuracy(model, data_val, 10, True)
print(f'Accuracy of the network after attacked: {tot_acc} %')
for i in range(10):
    print(f'Accuracy of {class_name[i]}: {idv_acc[i]} %')

