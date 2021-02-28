import torch
import io

file2 = open("test.txt", "w+")

count = 0
for i in range(512):
    for j in range(25):
        for k in range(25):
            tmp = str(k)
            if j == 24:
                file2.write(tmp)
                count += 1
            else:
                file2.write(tmp + ",")
                count += 1
        file2.write("!" + str(j) + "!")
    file2.write("\n")
print(count)
#file.write(dummy[0, :, :])