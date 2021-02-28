# Music Generator using simple RNN

import torch
import torch.nn as nn
import torch.optim as optim
import torch.functional as f
from scipy.io import wavfile as wf
import numpy as np
import matplotlib.pyplot as plt

# Audio data
w = wf.read('./sonata.wav')
length = len(w[1])
data = np.array(w[1])   # data shape = (12688128,2) = raw audio data
temp = data.transpose(1, 0) # Transposing data for easier access
Max = int(max((temp[0])))   # Max and Min peak value of the audio
Min = int(min((temp[0])))
data = torch.from_numpy(data)   # numpy to torch
data = data.to(dtype=torch.float32)
print(data.shape)
for i in range(length):
    if data[i][0] == 0:
        data[i][0] = 1

data = np.log(data[:][0])
print(data[:][0])
# Output space for output value
out_len = Max - Min + 1
out_space = np.linspace(Min, Max, out_len, dtype=np.int16)

'''
# Simple RNN using nn.rnn module

# Hyperparametes
num_classes = 5
input_size = 5  # one-hot size
output_size = out_len
batch_size = 100
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
'''