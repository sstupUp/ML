import numpy as np

data = [[1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]]

theta = np.radians(90)
c, s = np.cos(theta), np.sin(theta)
R = np.array(((c, -s), (s, c)))
ndata = np.zeros((3, 3))

for x in range(3):
    for y in range(3):
        tmp = np.zeros((2, 1))
        tmp = [[x], [y]]
        new = np.dot(R, tmp)
        new = np.round(new)
        n_x = int(new[0, 0])
        n_y = int(new[1, 0])

        ndata[n_x][n_y] = data[2 - x][2 - y]

print(new)
print(ndata)