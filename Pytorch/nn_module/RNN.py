# Simple RNN using nn.rnn module
# We teach our model to output 'ihello' with input 'hihell'

import torch
import torch.nn as nn
import torch.optim as optim
import torch.functional as f
import numpy as np

# Data and one-hot encoding

idx2char = ['h', 'i', 'e', 'l', 'o']
one_hot_lookup = [[1, 0, 0, 0, 0],
                  [0, 1, 0, 0, 0],
                  [0, 0, 1, 0, 0],
                  [0, 0, 0, 1, 0],
                  [0, 0, 0, 0, 1]]

x_data = [0, 1, 0, 2, 3, 3]  # hihell
val_data = [0, 4, 3, 3, 1, 4]   # hollio
y_data = [1, 0, 2, 3, 3, 4]  # ihello
x_one_hot = [one_hot_lookup[x] for x in x_data]
val_one_hot = [one_hot_lookup[x] for x in val_data]

inputs = torch.Tensor(x_one_hot)
labels = torch.LongTensor(y_data)


# Hyperparametes
num_classes = 5
input_size = 5  # one-hot size
output_size = 5
batch_size = 1
sequence_length = 6
num_layer = 1   # one layer RNN
learning_rate = 1e-1


# rnn model
class RNN(nn.Module):

    def __init__(self):
        super(RNN, self).__init__()
        self.rnn = nn.RNN(input_size=input_size, hidden_size=output_size, batch_first=True)
        self.fc = nn.Linear(output_size, num_classes)

    def forward(self, hidden, x):
        # Reshape input in (batch_size, sequence_length, input size)
        x = x.view(batch_size, sequence_length, input_size)
        out, hidden = self.rnn(x, hidden)
        return self.fc(out.view(-1, num_classes)), hidden_s

    def init_hidden_state(self):
        # initiate the hidden and cell state
        # (num_layers * num_directions, batch, output_size
        return torch.zeros(num_layer, batch_size, output_size)


fc = nn.Linear(output_size, num_classes)
model = RNN()
print(model)

# loss and optimizer
loss_fn = nn.CrossEntropyLoss()
optim_fn = optim.Adam(model.parameters(), lr=learning_rate)

tmp = []
# training the model
for epoch in range(200):
    loss = 0
    hidden_s = model.init_hidden_state()

    output, h_tmp = model(hidden_s, inputs)
    val, idx = output.max(1)
    result_str = [idx2char[c] for c in idx.squeeze()]

    loss = loss_fn(output, labels)

    if (epoch % 100) == 0:
        print("Predicted string: ", ''.join(result_str), end='')
        print(", epoch: %d, loss: %1.8f" % (epoch, loss))

    tmp = []
    optim_fn.zero_grad()
    loss.backward()
    optim_fn.step()

print(model.weight_ih_l0)
