import numpy
import matplotlib.pyplot as plt
import time, statistics


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


def compute_mandelbrotv2(xmin, xmax, ymin, ymax, width, height, max_iter, display):
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

N = 10000
A = numpy.random.rand(N, N)

# Quick: 0.1445s
def compute_row_sums(A):
    for i in range(N):
        s = numpy.sum(A[i, :])

# Slower than row sums: 0.4936s
def compute_column_sums(A):
    for j in range(N):
        s = numpy.sum(A[:, j])


A_f = numpy.asfortranarray(A)
# Slower than normal row-major: 0.5058s
def compute_asfortranarray_row_sums(A_f):
    for i in range(N):
        s = numpy.sum(A_f[i, :])

# Faster than norma column-major: 0.1402s
def compute_asfortranarray_column_sums(A_f):
    for j in range(N):
        s = numpy.sum(A_f[:, j])


def benchmark (func, *args, n_runs=3):
    """ Time func , return median of n_runs . """
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        result = func(*args )
        times.append(time.perf_counter() - t0)
        median_t = statistics.median(times)
        print(f"Median  {median_t:.4f}s"
        f"(min={min(times):.4f}, max={max(times):.4f})")
    return median_t, result


# -2, 1, -1.5, 1.5, 1024, 1024, 100, display=False
t, M = benchmark(compute_asfortranarray_row_sums, A_f)

#Milestone 4
# 256 = 0.0506s
# 512 = 0.3069s
# 1024 = 1.2093s
# 2048 = 4.6193s
# 4096 = 18.5580s