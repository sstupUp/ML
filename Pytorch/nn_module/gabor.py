import torch
import numpy as np
import matplotlib.pyplot as plt

ks = 100
theta = np.pi/2
sigma_x = 30
sigma_y = 30
frequency = np.pi/4
offset = np.pi/2

w = ks // 2
grid_val = torch.arange(-w, w+1, dtype=torch.float)
x, y = torch.meshgrid(grid_val, grid_val)
rotx = x * np.cos(theta) + y * np.sin(theta)
roty = -x * np.sin(theta) + y * np.cos(theta)
g = torch.zeros(y.shape)
g[:] = torch.exp(-0.5 * (rotx ** 2 / sigma_x ** 2 + roty ** 2 / sigma_y ** 2))
g /= 2 * np.pi * sigma_x * sigma_y
g *= torch.cos(2 * np.pi * frequency * rotx + offset)

plt.imshow(g)
plt.show()