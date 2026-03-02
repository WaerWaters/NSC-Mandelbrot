import numpy
import matplotlib.pyplot as plt
import time, statistics
import cProfile, pstats

# -2, 1, -1.5, 1.5, 1024, 1024, 100, display=False

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

# t, M = benchmark(compute_asfortranarray_row_sums, A_f)

"""
Lecture 2, Milestone 4
256 = 0.0506s
512 = 0.3069s
1024 = 1.2093s
2048 = 4.6193s
4096 = 18.5580s
"""

cProfile.run('compute_mandelbrot_naive(-2, 1, -1.5, 1.5, 1024, 1024, 100, display=False)', 'naive_profile.prof')
cProfile.run('compute_mandelbrot_numpy(-2, 1, -1.5, 1.5, 1024, 1024, 100, display=False)', 'numpy_profile.prof')

for name in ('naive_profile.prof', 'numpy_profile.prof'):
    stats = pstats.Stats(name)
    stats.sort_stats('cumulative')
    stats.print_stats(10)

"""
Lecture 3
Naive takes the most time to run
Some are called more than 20 million times
Still faster, fewer calls, has a bigger list than Naive
"""
"""
cProfile output
Mon Mar  2 12:17:20 2026    naive_profile.prof

         23046238 function calls (23043163 primitive calls) in 6.666 seconds

   Ordered by: cumulative time
   List reduced from 35 to 10 due to restriction <10>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    6.666    6.666 {built-in method builtins.exec}
        1    0.002    0.002    6.666    6.666 <string>:1(<module>)
        1    5.082    5.082    6.664    6.664 C:\Users\danie\NSC-Mandelbrot\mandelbrot.py:8(compute_mandelbrot_naive)
 21959734    1.481    0.000    1.481    0.000 {built-in method builtins.abs}
  1049600    0.064    0.000    0.064    0.000 {method 'append' of 'list' objects}
     1025    0.001    0.000    0.038    0.000 <__array_function__ internals>:177(linspace)
4100/1025    0.005    0.000    0.037    0.000 {built-in method numpy.core._multiarray_umath.implement_array_function}
     1025    0.012    0.000    0.036    0.000 C:\Users\danie\AppData\Local\Programs\Python\Python310\lib\site-packages\numpy\core\function_base.py:23(linspace)
     1025    0.001    0.000    0.012    0.000 <__array_function__ internals>:177(any)
     1025    0.001    0.000    0.010    0.000 C:\Users\danie\AppData\Local\Programs\Python\Python310\lib\site-packages\numpy\core\fromnumeric.py:2307(any)


Mon Mar  2 12:17:22 2026    numpy_profile.prof

         138 function calls (129 primitive calls) in 1.371 seconds

   Ordered by: cumulative time
   List reduced from 66 to 10 due to restriction <10>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    1.371    1.371 {built-in method builtins.exec}
        1    0.003    0.003    1.371    1.371 <string>:1(<module>)
        1    1.349    1.349    1.368    1.368 C:\Users\danie\NSC-Mandelbrot\mandelbrot.py:34(compute_mandelbrot_numpy)
     14/5    0.005    0.000    0.019    0.004 {built-in method numpy.core._multiarray_umath.implement_array_function}
        1    0.000    0.000    0.014    0.014 <__array_function__ internals>:177(meshgrid)
        1    0.000    0.000    0.014    0.014 C:\Users\danie\AppData\Local\Programs\Python\Python310\lib\site-packages\numpy\lib\function_base.py:4846(meshgrid)
        1    0.000    0.000    0.010    0.010 <__array_function__ internals>:177(broadcast_arrays)
        1    0.000    0.000    0.010    0.010 C:\Users\danie\AppData\Local\Programs\Python\Python310\lib\site-packages\numpy\lib\stride_tricks.py:480(broadcast_arrays)  
        1    0.009    0.009    0.009    0.009 C:\Users\danie\AppData\Local\Programs\Python\Python310\lib\site-packages\numpy\lib\stride_tricks.py:416(_broadcast_shape)  
        2    0.000    0.000    0.005    0.003 <__array_function__ internals>:177(copyto)
"""
