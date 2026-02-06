import numpy
import matplotlib.pyplot as plt

xmin, xmax = -2, 1
ymin, ymax = -1.5, 1.5
width, height = 500, 500
max_iter = 100

saved_iter = []

for x in numpy.linspace(xmin, xmax, width):
    temp = []
    for y in numpy.linspace(ymin, ymax, height):
        c = complex(x,y)
        z = 0
        for i in range(max_iter):
            z = z**2 + c
            if abs(z) > 2:
                temp.append(i)
                break
        else:
            temp.append(max_iter)
    saved_iter.append(temp)

plt.imshow(saved_iter)
plt.show()

