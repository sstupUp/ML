# Simple RNN using nn.rnn module
# We teach our model to output 'ihello' with input 'hihell'

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Data and one-hot encoding

idx2char = ['h', 'i', 'e', 'l', 'o']
one_hot_lookup = [[1, 0, 0, 0, 0],
                  [0, 1, 0, 0, 0],
                  [0, 0, 1, 0, 0],
                  [0, 0, 0, 1, 0],
                  [0, 0, 0, 0, 1]]

x_data = [0, 1, 0, 2, 3, 3]  # hihell
y_data = [1, 0, 2, 3, 3, 4]  # ihello
x_one_hot = [one_hot_lookup[x] for x in x_data]

inputs = torch.Tensor(x_one_hot)
labels = torch.LongTensor(y_data)


# Hyperparametes
num_classes = 5
input_size = 5  # one-hot size
output_size = 5
batch_size = 1
sequence_length = 1
num_layer = 1   # one layer RNN
learning_rate = 1e-4


# rnn model
class RNN(nn.Module):

    def __init__(self):
        super(RNN, self).__init__()
        self.rnn = nn.RNN(input_size=input_size, hidden_size=output_size, batch_first=True)

    def forward(self, hidden, x):
        # Reshape input in (batch_size, sequence_length, input size)
        x = x.view(batch_size, sequence_length, input_size)

        out, hidden = self.rnn(x, hidden)
        return hidden, out.view(-1, num_classes)

    def init_hidden_state(self):
        # initiate the hidden and cell state
        # (num_layers * num_directions, batch, output_size
        return torch.zeros(num_layer, batch_size, output_size)


model = RNN()
print(model)

# loss and optimizer
loss_fn = nn.CrossEntropyLoss()
optim_fn = optim.Adam(model.parameters(), lr=learning_rate)

tmp = []
# training the model
for epoch in range(30000):
    loss = 0
    hidden_s = model.init_hidden_state()

    for input, label in zip(inputs, labels):
        hidden_s, output = model(hidden_s, input)
        val, idx = output.max(1)
        tmp += idx2char[idx]
        loss += loss_fn(output, torch.LongTensor([label]))

    if (epoch % 1000) == 0:
        print(*tmp, end='', sep='')
        print(", epoch: %d, loss: %1.3f" % (epoch, loss))

    tmp = []
    optim_fn.zero_grad()
    loss.backward()
    optim_fn.step()
