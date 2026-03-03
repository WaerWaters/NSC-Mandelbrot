import numpy
import matplotlib.pyplot as plt

@profile
def compute_mandelbrot_naive(xmin, xmax, ymin, ymax, width, height, max_iter, display):
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
    
    if display:
        plt.imshow(saved_iter, cmap="hot")
        plt.title("Mandelbrot Set")
        plt.colorbar()
        plt.savefig("mandelbrot.png")
        plt.show()
   
    return saved_iter


def compute_mandelbrot_numpy(xmin, xmax, ymin, ymax, width, height, max_iter, display):
    x = numpy.linspace(xmin, xmax, width)
    y = numpy.linspace(ymin, ymax, height)
    X, Y = numpy.meshgrid(x, y)
    C = X + 1j*Y
    z = numpy.zeros_like(C)
    mask = numpy.ones(C.shape, dtype=bool)
    for i in range(max_iter):
        z[mask] = z[mask]**2 + C[mask]
        mask_now = numpy.abs(z) <= 2
        mask = mask_now

    if display:
        plt.imshow(mask, cmap="hot", extent=[xmin, xmax, ymin, ymax])
        plt.title("Mandelbrot Set")
        plt.colorbar()
        plt.savefig("mandelbrot.png")
        plt.show()
   
    return 