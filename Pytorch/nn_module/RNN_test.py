# Simple RNN using nn.rnn module
# model: rnn -> fc
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
d_test = [0, 4, 3, 3, 1, 0]  # hollih
x_one_hot = [one_hot_lookup[x] for x in x_data]
t_one_hot = [one_hot_lookup[x] for x in d_test]

inputs = torch.Tensor(x_one_hot)
labels = torch.LongTensor(y_data)

# Hyperparametes
num_classes = 5
input_size = 5  # one-hot size
output_size = 5
batch_size = 1
sequence_length = 1
num_layer = 1  # one layer RNN
learning_rate = 1e-2


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
        out = self.fc(out)
        return hidden, out.view(-1, num_classes)

    def init_hidden_state(self):
        # initiate the hidden and cell state
        # (num_layers * num_directions, batch, output_size
        return torch.randn(num_layer, batch_size, output_size)


model = RNN()
print(model)

# loss and optimizer
loss_fn = nn.CrossEntropyLoss()
optim_fn = optim.Adam(model.parameters(), lr=learning_rate)

tmp = []
# training the model
for epoch in range(1000):
    loss = 0
    hidden_s = model.init_hidden_state()

    for input, label in zip(inputs, labels):
        hidden_s, output = model(hidden_s, input)
        val, idx = output.max(1)
        tmp += idx2char[idx]
        loss += loss_fn(output, torch.LongTensor([label]))

    if (epoch % 100) == 0:
        print(*tmp, end='', sep='')
        print(", epoch: %d, loss: %1.3f" % (epoch, loss))

    tmp = []
    optim_fn.zero_grad()
    loss.backward()
    optim_fn.step()


# Get weight and bias of the model
# Linear layer weight and bias: fc.weight, fc.bias
# RNN layer weight and bias: all_weight = [input-hidden weight, hidden-hidden weight,
#                                          input-hidden bias, hidden-hidden bias]
temp = model.fc.weight
rnn_weight = model.rnn.all_weights[0][0]
rnn_bias = model.rnn.all_weights
print('Linear weight:', temp)
print('RNN input hidden weight:',rnn_weight)
print('RNN bias:', rnn_bias)

test = torch.Tensor(t_one_hot)

test_arr = []
for val in test:
    hidden_s, output = model(hidden_s, val)
    val, idx = output.max(1)
    test_arr += idx2char[idx]

print(test_arr)