import torch

x = torch.tensor(3.0)
w = torch.tensor(4.0, requires_grad=True)
b = torch.tensor(0.1, requires_grad=True)

y = w * x + b   # Basic form of Linear Regression
y.backward()

print('dy/dx:', x.grad) # Since x doesn't have requires_grad as True, x.grad is None
print('dy/dw:', w.grad) # dy/dw = x, which is 3.0
print('dy/db:', b.grad) # dy/db = 1
