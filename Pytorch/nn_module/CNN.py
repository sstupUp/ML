import torch
import torch.utils.data as util
import torchvision.datasets as dsets
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torch.nn.functional as f

# Load datasets(CIFAR10)
path = './'
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.49, 0.49, 0.49), (0.49, 0.49, 0.49))])

cifar10_t = dsets.CIFAR10(path, download=True, train=True, transform=transform)
cifar10_val = dsets.CIFAR10(path, download=True, train=False, transform=transform)
data = util.DataLoader(cifar10_t, batch_size=4, shuffle=True)
data_val = util.DataLoader(cifar10_val, batch_size=4, shuffle=True)

class_name = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Build a model
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        # input_channel, output_channel(# of filters), filter size
        self.conv1 = nn.Conv2d(3, 16, 5, padding=2)
        # pooling (2 x 2) with stride of 2
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 8, 3, padding=2)
        # Use the formula to get the input size
        self.fc1 = nn.Linear(9*9*8, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 10)

    def forward(self, x):
        x = self.pool(f.relu(self.conv1(x)))
        x = self.pool(f.relu((self.conv2(x))))
        x = x.view(-1, 9*9*8)
        x = f.relu(self.fc1(x))
        x = f.relu(self.fc2(x))
        x = self.fc3(x)

        return x



# Optimizer & loss
model = ConvNet()
learning_rate = 1e-4
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
loss = nn.CrossEntropyLoss()


# Training loop
def training_loop(n_epoch, network, optim_fn, loss_fn, data_t):
    for i in range(0, n_epoch + 1):
        for n, target in enumerate(data_t):
            img, label = target
            # img = img.to('cuda')
            # label = label.to('cuda')
            # forward
            prediction = network(img)
            loss_train = loss_fn(prediction, label)

            # backward
            optimizer.zero_grad()
            loss_train.backward()
            optimizer.step()

            if n % 1000 == 0:
                print(f"Epoch: {i}, Batch: {n}, Training loss {loss_train.item():.4f},")

            if (n >= 12000) & (n % 10 == 0):
                print(f"Epoch: {i}, Batch: {n}, Training loss {loss_train.item():.4f},")

# run
n_epochs = 3
training_loop(
    n_epoch=n_epochs,
    network=model,
    optim_fn=optimizer,
    loss_fn=loss,
    data_t=data)


print('Calculating Accuracy...')

device = 'cuda'
batch_size = 4

with torch.no_grad():
    n_correct = 0
    n_samples = 0
    n_class_correct = [0 for i in range(10)]
    n_class_samples = [0 for i in range(10)]
    for images, labels in data_val:
        # images = images.to(device)
        # labels = labels.to(device)
        outputs = model(images)
        # max returns (value ,index)
        _, predicted = torch.max(outputs, 1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()

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
