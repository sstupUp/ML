import torch

'''
# Torch basics
x = torch.tensor([[0, 0, 1], [0, 1, 0], [1, 0, 0]])
print(x)

x[1][1] = 5
print(x)

y = torch.zeros(2, 2, dtype=int)
print(y + torch.ones(2, 2, dtype=int))

# Tensors that live in the CPU and GPU
'''
'''
cpu_t = torch.rand(2)
print(cpu_t.device)

gpu_t = cpu_t.to("cuda")
print(gpu_t.device)
'''
'''

# Extracting the max value and its index
t = torch.rand(2,2)
print(t)
print('max value as tensor =', t.max(), 'max value as Python value = ', t.max().item(), 'Index =', t.argmax().item())

# Changing the type of a tensor using 'to()'
long_t = torch.tensor([[0, 1], [1, 0]])
print(long_t.type())
float_t = long_t.to(dtype=torch.float32)
print(float_t.type())

# Reshaping a tensor or other data using view() or reshape()
flat_tensor = torch.rand(784)
viewed = flat_tensor.view(1, 28, 28)    # 1 is for channel. 3 goes for 3 channels(RGB)
print(viewed.shape)
reshaped = flat_tensor.reshape(1, 28, 28)
print(reshaped.shape)

# Permuting a vector
hwc_tensor = torch.rand(640, 480, 3)    # [height, width, channel]
print(hwc_tensor.shape)
chw_tensor = hwc_tensor.permute(2, 0, 1)    # [channel, height, width]
print(chw_tensor.shape)

'''

# Scratches
bikes = torch.zeros(24, 17)
bikes[:1] = 1
print(bikes[0])


