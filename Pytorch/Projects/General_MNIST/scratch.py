import numpy as np
import torch
from scipy import ndimage

data = [[1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]]

theta = np.radians(90)
c, s = np.cos(theta), np.sin(theta)
R = np.array(((c, -s), (s, c)))
ndata = np.zeros((3, 3))

new_img = ndimage.rotate(data, 90, reshape=False)
new_img = ndimage.rotate(data, 90, reshape=False)
print(new_img)
