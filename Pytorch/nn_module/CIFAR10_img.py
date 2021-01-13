# We use CIFAR-10 data to make img classification network
# Distinguishing birds from airplanes
import torch
from torchvision import datasets
# We can transform a PIL(Python Imaging Library) image to a tensor
from torchvision import transforms

# Use datasets module to download CIFAR-10
# If train is set to false, it will download the validation set
data_path = './'
cifar10 = datasets.CIFAR10(data_path, train=True, download=True)
cifar10_val = datasets.CIFAR10(data_path, train=False, download=True)

# Transforming the PIL image to tensors
cifar10_t = datasets.CIFAR10(data_path, train=True, download=False, transform=transforms.ToTensor())
cifar10_t_val = datasets.CIFAR10(data_path, train=False, download=False, transform=transforms.ToTensor())

# Building the data set
class_names = ['AIRPLANE', 'BIRD']
label_map = {0: 0, 2: 1}
cifar2 = [(img, label_map[label]) for img, label in cifar10_t if label in [0, 2]]
cifar2_val = [(img, label_map[label])for img, label in cifar10_t_val if label in [0, 2]]
imgs = torch.stack([img_t for img_t, _ in cifar2], dim=3)
imgs_val = torch.stack([img_t for img_t, _ in cifar2_val], dim=3)

print(imgs.shape)
print(imgs_val.shape)

# Normalizing the images
img_mean = imgs.view(3, -1).mean(dim=1)    # We use view(3, -1) to turn 3*32*32 image into 3*1024 vector,
                                           # and calculate the mean over the 1024 elements of each channel
                                           # The value should be [0.4915, 0.4823, 0.4468]
img_std = imgs.view(3, -1).std(dim=1)      # The same goes with the standard deviation
# Norm = transforms.Normalize(img_mean, img_std)    # We can use the below functions, but for some reasons
# img_tn = Norm(imgs)                               # its not working
imgs_n = torch.empty_like(imgs)
for i in range(0, 3):
    imgs_n[i] = (imgs[i] - img_mean[i]) / img_std[i]

img_mean = imgs_val.view(3, -1).mean(dim=1)
img_std = imgs_val.view(3, -1).std(dim=1)
# Norm = transforms.Normalize(img_mean, img_std)
# img_t_val_n = Norm(imgs_val)
imgs_val_n = torch.empty_like(imgs_val)
for i in range(0, 3):
    imgs_val_n[i] = (imgs_val_n[i] - img_mean[i]) / img_std[i]

