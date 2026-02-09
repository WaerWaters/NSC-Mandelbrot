import numpy
import matplotlib.pyplot as plt
import time


def compute_mandelbrot(xmin, xmax, ymin, ymax, width, height, max_iter, display):
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
            temp.append(max_iter)
        saved_iter.append(temp)

    plt.imshow(saved_iter, cmap="hot")
    plt.title("Mandelbrot Set")
    plt.colorbar()
    plt.savefig("mandelbrot.png")
    if display:
        plt.show()
   
    return saved_iter


start = time.time()
results = compute_mandelbrot(-2, 1, -1.5, 1.5, 1024, 1024, 100, display=False)
elapsed = time.time() - start
print(f"computation took {elapsed:.3f} seconds")

