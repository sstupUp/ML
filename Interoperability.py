import numpy as np
import torch

x = np.array([[1, 2], [3, 4.]])
print(x, x.dtype)

y = torch.from_numpy(x) # .from_numpy() doesn't create a copy
print(y, y.dtype)

z = torch.tensor(x) # .tensor() creates a copy from x
print(z, z.dtype)

y[0][0] = -1
print('x: ', x)
print('y: ', y)

z[0][0] = -10
print('x: ', x)
print('z: ', z)