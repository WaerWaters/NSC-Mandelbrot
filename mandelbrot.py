import numpy
import matplotlib.pyplot as plt

xmin = -2
xmax = 1
ymin = -1.5
ymax = 1.5

width = 500
height = 500


complex_numbers = []

for i in numpy.linspace(xmin, xmax, width):
    temp = []
    for j in numpy.linspace(ymin, ymax, height):
        c = complex(i,j)
        temp.append(c)
    complex_numbers.append(temp)


saved_iter = []
max_iter = 100

for row in complex_numbers:
    temp = []
    for i in row:
        z = 0
        for j in range(max_iter):
            z = z**2 + i
            if abs(z) > 2:
                temp.append(j)
                break
        else:
            temp.append(max_iter)
    saved_iter.append(temp)

plt.imshow(saved_iter)
plt.show()

