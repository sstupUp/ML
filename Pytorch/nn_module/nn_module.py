# Deep Learning with Pytorch pg.156

import torch
import torch.nn as nn
import torch.optim as optim

# Data Set
t_c = [0.5, 14.0, 15.0, 28.0, 11.0, 8.0, 3.0, -4.0, 6.0, 13.0, 21.0]
t_u = [35.7, 55.9, 58.2, 81.9, 56.3, 48.9, 33.9, 21.8, 48.4, 60.4, 68.4]

t_c = torch.tensor(t_c)
t_u = torch.tensor(t_u)

# Making the data into one feature form
t_c = torch.unsqueeze(t_c, 1)
t_u = torch.unsqueeze(t_u, 1)
t_un = 0.1 * t_u

# Making linear model using nn.module
# Linear( #_of_features, the_size_of_output)
linear_model = nn.Linear(1, 1)
linear_model(t_un)

x = torch.ones(10, 1)
print(linear_model(x))

print(linear_model.weight)
print(linear_model.bias)
